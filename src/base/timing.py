import time

MAX_INT = 9223372036854775807

class TimedFunction(object):
  
  def __init__(self, max_time, max_steps):
    self.max_time = max_time if max_time > 0 else MAX_INT
    self.max_steps = max_steps if max_steps > 0 else MAX_INT
    pass
  
  def setup(self):
    pass
  
  def step(self):
    pass

  def finish(self):
    pass
  
  def run(self):
    start = time.time()
    self.setup()
    for(int i; i<=max_steps; i++):
      self.step()
      if end - start >= self.max_time:
        break
    return self.finish()
    
def create_timed_function(obj, setup, step, finish, max_time=0, max_steps=0):
  class ATimedFunction(TimedFunction):
    def __init__(self):
      self.obj = obj
    def setup(self):
      setup(self.obj)
    def step(self):
      step(self.obj)
    def finish(self):
      finish(self.obj)
  return ATimedFunction()

def run_timed_function(obj, setup, step, finish, max_time=0, max_steps=0):
  return create_timed_function(obj, setup, step, finish, max_time, max_steps)().run()
