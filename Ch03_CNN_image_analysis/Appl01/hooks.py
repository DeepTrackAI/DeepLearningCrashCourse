class fwd_hook():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)   
    def hook_func(self, m, i, o):
        print('Forward hook running...') 
        self.stored = o.detach().clone()
        print(f'Activations size: {self.stored.size()}')
    def __enter__(self, *args): 
        return self
    def __exit__(self, *args): 
        self.hook.remove()

class bwd_hook():
    def __init__(self, m):
        self.hook = m.register_full_backward_hook(self.hook_func)
    def hook_func(self, m, gi, go):
        print('Backward hook running...')
        self.stored = go[0].detach().clone()
        print(f'Gradients size: {self.stored.size()}')
    def __enter__(self, *args): 
        return self
    def __exit__(self, *args): 
        self.hook.remove()

# patch to have the module.register_full_backward_hook firing on the first module it was registered on.
from functools import wraps

def patch_first_hook(model):
    def set_requires_grad(arg):
        if torch.is_grad_enabled() and isinstance(arg, torch.Tensor):
            arg.requires_grad = True
        return arg

    old_forward = model.forward

    @wraps(old_forward)
    def wrapper(*args, **kwargs):
        return old_forward(
            *(set_requires_grad(arg) for arg in args),
            **{kw: set_requires_grad(arg) for kw, arg in kwargs.items()})

    setattr(model, 'forward', wrapper)