from setuptools import setup, find_packages

setup(name="psych_dashboard",
      version='0.0.1',
      description='A package to run a Plotly Dash dashboard to illustrate data from single data files',
      install_requires=['pandas>=1.0.0',
                        'xlrd>=1.2.0',
                        'plotly>=4.8.0',
                        'dash>=1.12.0',
                        'scikit-learn>=0.22.0',
                        'numpy>=1.16.0',
                        'scipy>=1.2.0',
                        'feather-format>=0.4.0',
                        'dash-bootstrap-components>=0.10.0',
                        'selenium>=3.0.0',
                        'reportlab>=3.5.50',
                        ],
      entry_points={
         'console_scripts': ['run-dashboard=psych_dashboard.index:main'],
        },
      packages=find_packages(where='src'),
      package_dir={"": "src"},
      )
