# -*- coding: utf-8 -*-

"""
Add Spinbox widget to ttk (better looking themes).

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import sys
if sys.version_info[0] < 3:
    import ttk
else:
    import tkinter.ttk as ttk
    
__all__ = [ "Spinbox" ]


class Spinbox(ttk.Entry):
    
    def __init__(self, master = None, **kwargs):
        ttk.Entry.__init__(self, master, "ttk::spinbox", **kwargs)
        
    def current(self, newindex = None):
        return self.tk.call(self._w, "current", newindex)
    
    def set(self, value):
        return self.tk.call(self._w, "set", value)