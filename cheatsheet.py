lis = [1, 2, 3, 4, 5, 6]
dir(lis)
for elem in lis: 
  print(elem)

it = iter(lis)
it = lis.__iter__()
next(it)

# List comprehension -- returns list
[x*x for x in lis]
[x*x for x in iter(lis)]
iter(lis).reduce()

# Generator expression -- returns generator (which differs from an iterator how?)
(x*x for x in lis)            # => gen
list( (x*x for x in lis) )    # create list from gen
list( (x,y) for x in lis if x%2 == 0 for y in lis if y%2 == 1) # n x m matrix

# generator func
def genfun():
  for val in [3,7,12]:
    yield val

list(genfun())

# https://docs.python.org/2/howto/functional.html: "Passing values into a generator" -- weird feature at 1st glance!
# So there's iter.send(val), not just iter.next()!
# Where would we use that feature meaningfully?
# http://stackoverflow.com/questions/2776829/difference-between-python-generators-vs-iterators 
# "Every generator is an iterator, but not vice versa."
#iter(lis).send(5) # no such thing
def genfun():
  val = 10
  while val >= 0:
    sentValue = (yield val)  # sentValue is the param to send(sentValue), or None for generator.next() aka send(None)
    val -= sentValue if sentValue != None else 1;

# python xrange deprecated in python 3: http://www.pythoncentral.io/how-to-use-pythons-xrange-and-range/

# Two of Python�s built-in functions, map() and filter(), are somewhat obsolete; they duplicate the features
# of list comprehensions but return actual lists instead of iterators.
list(map(lambda x: x+10, genfun()))
list(filter(lambda x: x%2==0, genfun()))
# same funcs as generator expressions
list(x+10 for x in genfun())
list(x for x in genfun() if x%2==0)

# Reduce: see also http://stackoverflow.com/questions/181543/what-is-the-problem-with-reduce
import functools # The reduce function, since it is not commonly used, was removed from the built-in functions in Python 3.
functools.reduce(lambda accu, val: accu + val, genfun())
# compare with itertools:
import itertools
list(itertools.accumulate(genfun(), lambda accu, val: accu + val)) # aka reduce with args swapped, note it returns an iter though

list(genfun())
g = genfun()
print(next(g))
print(next(g))
print(next(g))
print(g.send(None))
print(g.send(0))
print(g.send(0))
print(g.send(0))
print(g.send(2))
print(g.send(2))


dic = {"a": 7, "b": "hey"}
for key in dic: 
  print(key)

# custom iterator
  
class MyIter(object):
  def __init__(self, source):
    self.source = source
  def __next__(self):
    nextItem = next(self.source) # or self.source.__next__()
    self.lastItem = nextItem
    return nextItem
  def __iter__(self):
    return self

[x for x in MyIter([1,3,5].__iter__())]
list(MyIter([1,3,5].__iter__()))
list(MyIter(iter([1,3,5])))


# GeneratorExit & generator.close()
def mygen():
  try:
    for i in range(0,10):
      yield i
  except GeneratorExit: # optional
    print("caught GeneratorExit");

list(mygen())

gen = mygen()
print(next(gen))
print(next(gen))
gen.close()


# couroutine with send()
# see also http://legacy.python.org/dev/peps/pep-0342/, which makes send() the feature that distinguishes coro from gen
# "Calling send(None) is exactly equivalent to calling a generator's next() method"
def myco():
  for i in range(0,10):
    print("myco", i)
    yield

co = myco()
next(co)
next(co)
co.send(None)


# yield from, which just 'flattens' the stream of returned values, see http://legacy.python.org/dev/peps/pep-0380/
def myco():
  for i in range(0,5):
    print("myco", i)
    yield i
    
def outerCo():
  yield from myco()
  yield from myco()

list(outerCo())
  
# see https://docs.python.org/2/howto/functional.html
import functools # partial
import math
def max4(x):
  return max(4, x);
# or 
max4 = lambda x: max(4, x)
# or 
max4 = functools.partial(max, 4) # like lodash.js bind
print(max4(2), max4(6))

# kwargs
def fun(a, b=5, c=6):
  print(a,b,c)

fun(1, c=8)

def start(*args,**kwargs):
  print("before")
  fun(*args,**kwargs)
  print("after")

start(1, c=9)


# threading
import threading
import functools
import time
def fun(x):
  for i in range(0, x):
    time.sleep(0.2)
    print(i)

threading.Thread(target=lambda: fun(3)).start()


# Tasks are Futures
import asyncio
import time

@asyncio.coroutine
def coro(x):
  for i in range(0, x):
    yield from asyncio.sleep(0.2)
    print("coro ", i)
  return x*x

def fun(x):
  for i in range(0, x):
    time.sleep(0.1)
    print("fun", i)
  return 42

loop = asyncio.get_event_loop()
loop.run_until_complete(coro(4))
task = asyncio.async(coro(4))
loop.run_until_complete(asyncio.async(lambda: fun(4)))  # fails: fun is not a coro
loop.run_until_complete(asyncio.async(asyncio.coroutine(lambda: fun(4))()) ) # works, but syntax is horrible

def makeAsync(func):
  return asyncio.async(asyncio.coroutine(func)())

loop.run_until_complete(makeAsync(lambda: fun(4))) # decent syntax

# like std::async, running on threadpool
#   future = loop.run_in_executor(None, lambda: fun(5))
def coro2(x):
  for i in range(0, x):
    funResult = yield from loop.run_in_executor(None, lambda: fun(2))
    print("fun done", i,funResult)
loop.run_until_complete(coro2(2))

# asyncio.gather = asyncio.wait = Q.all
@asyncio.coroutine
def testGather(x):
  parallelResults = yield from asyncio.gather(loop.run_in_executor(None, lambda: fun(4)), loop.run_in_executor(None, lambda: fun(4)))
  print(list(result))
  return parallelResults

result = loop.run_until_complete(testGather(2))

# asyncio.gather is not taking a list? That threw me off, note the * arg,
# see http://agiliq.com/blog/2012/06/understanding-args-and-kwargs/

@asyncio.coroutine
def testGather(x):
  parallelTasks = [loop.run_in_executor(None, lambda: fun(4)), loop.run_in_executor(None, lambda: fun(4))]
  parallelResults = yield from asyncio.gather(*parallelTasks) # note the funky *!
  print(list(parallelResults))
  return parallelResults

result = loop.run_until_complete(testGather(2))

@asyncio.coroutine
def testWait(x):
  parallelTasks = [loop.run_in_executor(None, lambda: fun(4)), loop.run_in_executor(None, lambda: fun(4))]
  parallelResults = yield from asyncio.wait(parallelTasks)
  print(list(parallelResults))
  # so wait returns futures, it doesn't unpack them
  return parallelResults

parallelResults = loop.run_until_complete(testWait(3))
parallelResults = [res.result() for res in (parallelResults[:-1])] # drop funky trailing set
print(list(parallelResults))

@asyncio.coroutine
def testWait(x):
  parallelTasks = (loop.run_in_executor(None, lambda: fun(4)) for i in range(0,x))
  parallelResults = yield from asyncio.wait(list(parallelTasks)) # the list is necessary
  print(list(parallelResults))
  return parallelResults

result = loop.run_until_complete(testWait(3))
print(result)

# lock RAII in python with 'with': https://docs.python.org/2/library/threading.html
import threading
some_rlock = threading.RLock()
with some_rlock:
    print "some_rlock is locked while this executes"
    
# using RAII in general:
class controlled_execution:
  def __enter__(self):
      print("---enter")
      return None
  def __exit__(self, type, value, traceback):
      print("---exit")

with controlled_execution():
   print("bla")
    
    
# python decorators, see http://simeonfranklin.com/blog/2012/jul/1/python-decorators-in-12-steps/
def mydecorator(function):
  print("---mydecorator exec")
  def inner(*args, **kw):
      print("---mydecorator inner exec")
      retval = function(*args, **kw)
      print("---mydecorator inner exit")
      return retval
  return inner
  
@mydecorator
def fun(x):
  return x+1

decorated(3)

# same as:
#decorated = mydecorator(fun)
#decorated(3)

# see also https://docs.python.org/2/library/contextlib.html for even simpler decorator syntax
import time
from contextlib import contextmanager

@contextmanager
def timethis(label):
  start = time.time()
  try:
    yield
  finally:
    end = time.time()
    print('%s: %0.3f' % (label, end-start))

with timethis("bla"):
  print("hello")

# another example with yield expr

import tempfile
import shutil
from contextlib import contextmanager
@contextmanager
def tempdir():
 outdir = tempfile.mkdtemp()
 try:
  yield outdir
 finally:
  shutil.rmtree(outdir)

with tempdir() as dirname:
  print(dirname)

# http://www.dabeaz.com/ 3 slide sets on generators & coroutines are great!

# unit tests http://docs.python-guide.org/en/latest/writing/tests/

# http://current.blogspot.ca/2014/09/private-variables-in-python.html