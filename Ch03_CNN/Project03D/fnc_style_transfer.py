
class fwd_hooks():
    def __init__(self, ms):
        self.ms = ms
        self.hook, self.stored = [], []
        for m in self.ms:
            self.hook.append(m.register_forward_hook(self.hook_func))
    def hook_func(self, m, i, o):
        # print('Forward hook running...') 
        self.stored.append(o)
        # print(f'Activations size: {o.size()}')
    def __enter__(self, *args): 
         return self
    def __exit__(self, *args): 
        for h in self.hook:    
            h.remove()