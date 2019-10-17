import torch
import pyro

# Introduction to Models in Pyro
pyro.set_rng_seed(101)

def first_steps():
    loc = 0. # zero mean
    scale = 1. # unit variance
    normal = torch.distributions.Normal(loc, scale)
    x = normal.rsample() # draw sample from N(0,1)
    print("sample", x)
    print("log prob", normal.log_prob(x)) # score the sample


    y = pyro.sample("my_sample", pyro.distributions.Normal(loc, scale))
    print(y)

# joint probability distribution over two random variables: cloudy and temp
def weather_pyro():
    # 30% of the time it's cloudy
    cloudy = pyro.sample('cloudy', pyro.distributions.Bernoulli(0.3))
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
    temp = pyro.sample('temp', pyro.distributions.Normal(mean_temp, scale_temp))
    return cloudy, temp.item()

for _ in range(3):
    print(weather_pyro())

def ice_cream_sales():
    cloudy, temp = weather()
    expected_sales = 200. if cloudy == 'sunny' and temp > 80.0 else 50.
    ice_cream = pyro.sample('ice_cream', pyro.distributions.Normal(expected_sales, 10.0))
    return ice_cream

print(ice_cream_sales())

def geometric(p, t=None):
    if t is None:
        t = 0
    x = pyro.sample("x_{}".format(t), pyro.distributions.Bernoulli(p))
    print(x)
    if x.item() == 1:
        return 0
    else:
        return 1 + geometric(p, t + 1)

print(geometric(0.5))

def normal_product(loc, scale):
    z1 = pyro.sample("z1", pyro.distributions.Normal(loc, scale))
    z2 = pyro.sample("z2", pyro.distributions.Normal(loc, scale))
    y = z1 * z2
    return y

#takes one argument 'scale=1.' and generates three random variables z1, z2 and mu_latent
def make_normal_normal():
    mu_latent = pyro.sample("mu_latent", pyro.distributions.Normal(0, 1))
    fn = lambda scale: normal_product(mu_latent, scale)
    return fn

print(make_normal_normal()(1.))


