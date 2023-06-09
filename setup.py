import sys
from setuptools import setup


def main():
    if sys.version_info.major < 3 or sys.version_info.minor < 10:
        print(f'This Python is compatible with Python 3.10 and above, but you are '
              f'running Python {sys.version_info.major}.{sys.version_info.minor}. The installation will likely fail.') 
    
    extras = {
        'test': [
            'pytest>=7.3.1',
            'coverage>=7.2.7',
        ],
        'tf2': [
            'tensorflow>=2.12.0',
        ],
        'analysis': [
            'matplotlib>=3.7.1',
            'mplfinance>=0.12',
            'jupyter>=1.0.0',
        ],
        'ray': [
            'ray[all]=>2.5.0',
        ],
        'sb3': [
            'stable-baselines3>=1.8.0',
        ]
    }
    all_extras = []
    for group_name in extras:
        all_extras += extras[group_name]

    extras['all'] = all_extras

    setup(
        name='drl-investment',
        install_requires=[
            'torch>=2.0.1',
            'numpy>=1.24.3',
            'pandas>=2.0.2',
            'gymnasium>=0.28.1'
        ],
        extras_require=extras,
        description='drl-investment: investment tool base on deep reinforce learning',
        author='lishuliang',
        url='https://github.com/l1351868270/DRL-investment',
        author_email='llishuliang@163.com',
        version='0.0.1',
    )


if __name__ == '__main__':
    main()
