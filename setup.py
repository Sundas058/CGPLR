from setuptools import setup, find_packages


setup(name='cgplr-reid',
      version='1.0.0',
      description='Confidence-Guided Pseudo-Label Refinement for Unsupervised Cross-Domain Person Re-Identification',
      author='Sundas Iqbal, Danish Ali, Qing Tian, Jianping Gou, Weihua Ou',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss-gpu'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Unsupervised Domain Adaptation',
          'Pseudo-Label Refinement',
          'Confidence-aware Learning',
          'Person Re-identification'
      ])
