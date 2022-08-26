import warnings
def f():
     print('before')
     warnings.warn('you are warned!')
     print('after')

f()

warnings.filterwarnings("ignore")

f()