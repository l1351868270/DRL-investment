from setuptools import setup


project_name = 'drl-investment'


def main():
    setup(
        name=project_name,
        version='0.0.1',
        install_requires=['pytest'],
    )


if __name__ == '__main__':
    main()
