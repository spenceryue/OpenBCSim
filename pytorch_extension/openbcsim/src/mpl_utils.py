# from contextlib import contextmanager


# @contextmanager
# def rcParams (new_styles: dict, **kwargs):
#   from matplotlib import rcParams as params
#   save = params.copy ()
#   try:
#     params.update (new_styles)
#     kwargs.pop ('new_styles', None)
#     params.update (kwargs)
#     yield
#   except Exception as e:
#     raise e
#   finally:
#     params.update (save)


from contextlib import ContextDecorator


class rcParams (ContextDecorator):
  def __init__ (self, new_styles: dict, **kwargs):
    from matplotlib import rcParams as params
    self.save = params.copy ()
    self.params = params
    self.new_styles = new_styles
    self.kwargs = kwargs
    self.kwargs.pop ('new_styles', None)

  def __enter__ (self):
    self.params.update (self.new_styles)
    self.params.update (self.kwargs)
    return self.params

  def __exit__ (self, *exc):
    self.params.update (self.save)
