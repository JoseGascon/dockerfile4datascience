from datetime import datetime
import json
import io

from invoke import task
import boto3


@task
def pex(ctx):
    ctx.run('pex --python-shebang="#!/usr/bin/env python" --no-use-wheel -r requirements.txt -c invoke -o inv.pex')


@task
def create_machine(ctx, name, profile):
    cmd = " ".join([
        "docker-machine create",
        "--driver amazonec2",
        # "--amazonec2-ami ami-32cc1953",
        "--amazonec2-region ap-northeast-1",
        "--amazonec2-instance-type g2.2xlarge",
        name
    ])
    ctx.run(cmd, echo=True, env={'AWS_PROFILE': profile})


@task
def install_nvidia_docker(ctx, name):
    def __ssh(cmd):
        return ctx.run('docker-machine ssh {name} {cmd}'.format(name=name, cmd=cmd), echo=True)

    nvidia_version = "367.44"
    driver_url = "http://jp.download.nvidia.com/XFree86/Linux-x86_64/{ver}/NVIDIA-Linux-x86_64-{ver}.run".format(ver=nvidia_version)

    __ssh('"sudo apt-get update && sudo apt-get -y upgrade && sudo apt-get install --no-install-recommends -y gcc make libc-dev"')
    __ssh("wget -P /tmp {url}".format(url=driver_url))
    __ssh("sudo sh /tmp/NVIDIA-Linux-x86_64-{ver}.run --silent".format(ver=nvidia_version))

    # Install nvidia-docker and nvidia-docker-plugin
    __ssh("wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.0-rc.3/nvidia-docker_1.0.0.rc.3-1_amd64.deb")
    __ssh('"sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb"')

    nvidia_cli(ctx, name)


@task
def iid_region(ctx, name):
    result = ctx.run('docker-machine inspect {}'.format(name), hide='out')
    inspect = json.loads(result.stdout)
    driver = inspect['Driver']
    region = driver['Region']
    instance_id = driver['InstanceId']
    # print((instance_id, region))
    return (instance_id, region)


@task
def is_machine_stopped(ctx, name):
    result = ctx.run("docker-machine status {}".format(name), hide='out')
    return result.stdout == 'Stopped\n'


@task
def create_ami(ctx, name, profile):
    # prepare
    instance_id, region = iid_region(ctx, name)
    session = boto3.Session(profile_name=profile)

    ec2 = session.resource('ec2', region_name=region)
    instance = ec2.Instance(instance_id)
    client = session.client('ec2', region_name=region)
    waiter = client.get_waiter('image_available')

    # stop machine if necessary
    if not is_machine_stopped(ctx, name):
        ctx.run("docker-machine stop {name}".format(name=name))

    ami_name = 'aws-gpu-{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print('Creating ami "{}" ...'.format(ami_name))
    image = instance.create_image(
        Name=ami_name,
        Description='ami for tensorflow with gpu',
        NoReboot=True
        )
    waiter.wait(ImageIds=[image.image_id])
    print('Image "{}" was created.'.format(ami_name))
    print('ImageId="{}"'.format(image.image_id))

    # start machine if necessary
    ctx.run("docker-machine start {}".format(name))
    ctx.run("docker-machine regenerate-certs {}".format(name))


@task
def nvidia_cli(ctx, name):
    ctx.run("docker-machine ssh {name} curl -s http://localhost:3476/docker/cli".format(name=name))


@task
def build_docker_image(ctx, name):
    ctx.run("docker build -t rkawajiri/dockerfile4datascience:tensorflow_gpu_ec2 .", echo=True)
    ctx.run("docker push rkawajiri/dockerfile4datascience:tensorflow_gpu_ec2", echo=True)
