import math
import svgwrite
import pdb

def drange(start, stop, step):
  r = start
  while r < stop:
    yield r
    r += step

def gaussian(x, mu, sig):
  #matplotlib/numpy: return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
  norm = (1 / (sig * math.sqrt(2 * math.pi)))
  return math.exp(-math.pow(x - mu, 2.) / (2 * pow(sig, 2.))) * norm


#for x in drange(-3, 3, 0.1):
#  print(x, gaussian(x, 0, 1))

# PDF p(x|w1)
def p_x_w1(x):
  return gaussian(x, -1, 1)
def p_x_w2(x):
  return gaussian(x, 2, 1)
p_x_wi = [p_x_w1, p_x_w2]

# a priori P(w1)
P_w1 = 0.2
P_w2 = 1 - P_w1
P_wi = [P_w1, P_w2]

# bayes p(x)
def p_x(x):
  return sum(p_x_wi[i](x) * P_wi[i] for i in range(0,2))

def integral(f, fromX, toX, delta):
  integral = 0
  for x in drange(fromX, toX, delta):
    y = f(x)
    integral += y * delta
  return integral

print("integral p_x_w1", integral(p_x_w1, -5, 5, 0.1))
print("integral p_x_w2", integral(p_x_w2, -5, 5, 0.1))
print("integral p_x   ", integral(p_x   , -5, 5, 0.1))

# bayes P(w1|x)
def P_w1_x(x):
  i = 0
  return p_x_wi[0](x) * P_wi[0] / p_x(x)


svg_document = svgwrite.Drawing(filename = "test-svgwrite.svg",
                                size = ("800px", "600px"))

def map_coords(x, y):
  return (400 + x * 100, 600 - y * 600)

delta = 0.01
for x in drange(-5, 5, delta):
  y = p_x_w1(x)
  svg_document.add(svg_document.circle(center=map_coords(x,y), stroke='blue'))

  y = p_x_w2(x)
  svg_document.add(svg_document.circle(center=map_coords(x,y), stroke='red'))

  y = p_x(x)
  svg_document.add(svg_document.circle(center=map_coords(x,y), stroke='green'))

  y = P_w1_x(x)
  svg_document.add(svg_document.circle(center=map_coords(x,y)))
  
svg_document.save()

  