# -*- coding: utf-8 -*-

"""
StochOPy Viewer is a GUI for StochOPy to see how popular stochastic algorithms
perform on different benchmark functions.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from matplotlib.figure import Figure
from matplotlib import animation

import numpy as np
from ..evolutionary_algorithm import Evolutionary
from ..monte_carlo import MonteCarlo
from ..benchmark_functions import BenchmarkFunction

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog as tkfile
    import tkMessageBox as tkmessage
    import ttk
    import tkFont as font
else:
    import tkinter as tk
    import tkinter.filedialog as tkfile
    import tkinter.messagebox as tkmessage
    import tkinter.ttk as ttk
    from tkinter import font
from .ttk_spinbox import Spinbox
    
try:
    import cPickle as pickle
except ImportError:
    import pickle

__all__ = [ "StochOGUI", "main" ]


class StochOGUI():
    """
    GUI for StochOPy.
    
    StochOPy Viewer provides a GUI to manipulate solvers on popular benchmark
    functions.
    
    Parameters
    ----------
    master : tkinter object
        tkinter root window.
    """
    
    master = None
    anim_running = False
    first_run = True
    
    MAX_SEED = 999999
    FUNCOPT = ( "Ackley", "Quartic", "Quartic noise", "Rastrigin", "Rosenbrock",
                "Sphere", "Styblinski-Tang" )
    EAOPT = ( "CPSO", "PSO", "DE", "CMAES", )
    MCOPT = ( "Hastings", "Hamiltonian", )
    STRATOPT = ( "rand1", "rand2", "best1", "best2" )
    MIN_POPSIZE = { "cpso": 2, "pso": 2, "de": 4, "cmaes": 4 }
    
    def __init__(self, master):
        self.master = master
        master.title("StochOPy Viewer")
        master.protocol("WM_DELETE_WINDOW", self.close_window)
        master.geometry("900x600")
        master.minsize(900, 600)
        master.maxsize(900, 600)
        
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(family = "Helvetica", size = 9)
        master.option_add("*Font", default_font)
        
        self.define_variables()
        self.trace_variables()
        self.init_variables()
        self.menubar()
        self.frame1()
        self.frame2()
        self.footer()
        self.select_widget(self.solver_name.get())
        
    def about(self):
        about = "StochOPy Viewer 1.1" + "\n" \
                + "Created by Keurfon Luu"
        tkmessage.showinfo("About", about)
        
    def menubar(self):
        menubar = tk.Menu(self.master)
        
        # File
        filemenu = tk.Menu(menubar, tearoff = 0)
        filemenu.add_command(label = "Export models", command = self.export_models)
        filemenu.add_command(label = "Export fitness", command = self.export_fitness)
        filemenu.add_separator()
        filemenu.add_command(label = "Exit", command = self.close_window)
        
        # Help
        helpmenu = tk.Menu(menubar, tearoff = 0)
        helpmenu.add_command(label = "About", command = self.about)
        
        # Display menu bar
        menubar.add_cascade(label = "File", menu = filemenu)
        menubar.add_cascade(label = "Help", menu = helpmenu)
        self.master.config(menu = menubar)
        
    def frame1(self):
        self.frame1 = ttk.LabelFrame(self.master, text = "Parameters", borderwidth = 2, relief = "groove")
        self.frame1.place(bordermode = "outside", relwidth = 0.99, relheight = 0.21, relx = 0, x = 5, y = 5, anchor = "nw")
        self.frame1.first_run = True
        
        # function
        function_label = ttk.Label(self.frame1, text = "Function")
        function_option_menu = ttk.OptionMenu(self.frame1, self.function, self.function.get(),
                                              *sorted(self.FUNCOPT))
        
        # max_iter
        max_iter_label = ttk.Label(self.frame1, text = "Maximum number of iterations")
        max_iter_spinbox = Spinbox(self.frame1, from_ = 2, to_ = 9999,
                                   increment = 1, textvariable = self.max_iter,
                                   width = 6, justify = "right", takefocus = True)
        
        # fps
        fps_label = ttk.Label(self.frame1, text = "Delay between frames (ms)")
        fps_spinbox = Spinbox(self.frame1, from_ = 1, to_ = 1000,
                              increment = 1, textvariable = self.interval,
                              width = 6, justify = "right", takefocus = True)
        
        # seed
        seed_button = ttk.Checkbutton(self.frame1, text = "Fix seed",
                                      variable = self.fix_seed, takefocus = False)
        seed_spinbox = Spinbox(self.frame1, from_ = 0, to_ = self.MAX_SEED,
                               increment = 1, textvariable = self.seed,
                               width = 6, justify = "right", takefocus = True)
        
        # solver
        solver_label = ttk.Label(self.frame1, text = "Solver")
        
        solver_option_menu = ttk.OptionMenu(self.frame1, self.solver_name, self.solver_name.get(),
                                            *(self.EAOPT + self.MCOPT), command = self.select_widget)
        
        # constrain
        constrain_button = ttk.Checkbutton(self.frame1, text = "Constrain",
                                      variable = self.constrain, takefocus = False)
        
        # Layout
        function_label.place(relx = 0., x = 5, y = 5, anchor = "nw")
        function_option_menu.place(relx = 0., x = 75, y = 3, anchor = "nw")
        max_iter_label.place(relx = 0., x = 5, y = 30, anchor = "nw")
        max_iter_spinbox.place(width = 80, relx = 0., x = 220, y = 30, anchor = "nw")
        fps_label.place(relx = 0., x = 5, y = 55, anchor = "nw")
        fps_spinbox.place(width = 80, relx = 0., x = 220, y = 55, anchor = "nw")
        seed_button.place(relx = 0., x = 5, y = 80, anchor = "nw")
        seed_spinbox.place(width = 80, relx = 0., x = 220, y = 80, anchor = "nw")
        solver_label.place(relx = 0.35, x = 0, y = 5, anchor = "nw")
        solver_option_menu.place(relx = 0.35, x = 50, y = 3, anchor = "nw")
        constrain_button.place(relx = 0.35, x = 0, y = 80, anchor = "nw")
        
    def frame1_pop(self):
        if not self.frame1.first_run:
            self.frame1.pop.forget()
        self.frame1.pop = ttk.Frame(self.frame1, borderwidth = 0)
        self.frame1.pop.place(width = 170, height = 25, relx = 0.35, y = 30, anchor = "nw")
        if self.solver_name.get() in self.EAOPT:
            # popsize
            popsize_label = ttk.Label(self.frame1.pop, text = "Population size")
            popsize_spinbox = Spinbox(self.frame1.pop, from_ = 1, to_ = 999,
                                      increment = 1, textvariable = self.popsize,
                                      width = 3, justify = "right", takefocus = True)
            
            # Layout
            popsize_label.place(relx = 0, x = 0, y = 0, anchor = "nw")
            popsize_spinbox.place(width = 60, relx = 0, x = 110, y = 0, anchor = "nw")
            
    def frame1_sync(self):
        if not self.frame1.first_run:
            self.frame1.sync.forget()
        self.frame1.sync = ttk.Frame(self.frame1, borderwidth = 0)
        self.frame1.sync.place(width = 170, height = 25, relx = 0.35, y = 55, anchor = "nw")
        if self.solver_name.get() in [ "CPSO", "PSO", "DE" ]:
            # sync
            sync_button = ttk.Checkbutton(self.frame1.sync, text = "Synchronize",
                                          variable = self.sync, takefocus = False)
            
            # Layout
            sync_button.place(relx = 0, x =0, y = 0, anchor = "nw")
            
    def frame2(self):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        self.frame2 = ttk.Frame(self.master, borderwidth = 2, relief = "groove")
        self.frame2.place(bordermode = "outside", relwidth = 0.99, relheight = 0.72, relx = 0, rely = 0.22, x = 5, y = 5, anchor = "nw")
        self.frame2.canvas = ttk.Frame(self.frame2, borderwidth = 0)
        self.frame2.canvas.place(relwidth = 1, relheight = 1, relx = 0, anchor = "nw")
        self.fig = Figure(figsize = (13, 6), facecolor = "white")
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.frame2.canvas)
        self.fig.canvas.mpl_connect("button_press_event", self._onClick)
        self.canvas.get_tk_widget().pack()
        
    def init_widget(self):
        if not self.frame1.first_run:
            self.frame1.sliders.forget()
        else:
            self.frame1.first_run = False
        self.frame1.sliders = ttk.Frame(self.frame1, borderwidth = 0)
        self.frame1.sliders.place(relwidth = 0.45, relheight = 1., relx = 0.55, anchor = "nw")
        
    def select_widget(self, solver):
        self.frame1_pop()
        self.frame1_sync()
        if solver == "CPSO":
            self.cpso_widget()
        elif solver == "PSO":
            self.pso_widget()
        elif solver == "DE":
            self.de_widget()
        elif solver == "CMAES":
            self.cmaes_widget()
        elif solver == "Hastings":
            self.hastings_widget()
        elif solver == "Hamiltonian":
            self.hamiltonian_widget()
            
    def _label(self, text, position, kwargs = {}):
        label = ttk.Label(self.frame1.sliders, text = text, **kwargs)
        if position == 1:
            label.place(relx = 0, x = 0, y = 5, anchor = "nw")
        elif position == 2:
            label.place(relx = 0, x = 0, y = 50, anchor = "nw")
        elif position == 3:
            label.place(relx = 0.5, x = 0, y = 5, anchor = "nw")
        elif position == 4:
            label.place(relx = 0.5, x = 0, y = 50, anchor = "nw")
        return label
            
    def _scale(self, from_, to, resolution, variable, position, command = None,
               kwargs = {}):
        if command is None:
            command = lambda s: variable.set(round(float(s), 3))
        scale = ttk.Scale(self.frame1.sliders, from_ = from_, to = to, variable = variable,
                          orient = "horizontal", length = 20, command = command,
                          takefocus = False, **kwargs)
        if position == 1:
            scale.place(relwidth = 0.35, relx = 0, x = 0, y = 25, anchor = "nw")
        elif position == 2:
            scale.place(relwidth = 0.35, relx = 0, x = 0, y = 70, anchor = "nw")
        elif position == 3:
            scale.place(relwidth = 0.35, relx = 0.5, x = 0, y = 25, anchor = "nw")
        elif position == 4:
            scale.place(relwidth = 0.35, relx = 0.5, x = 0, y = 70, anchor = "nw")
        return scale
    
    def _entry(self, variable, position, kwargs = {}):
        entry = ttk.Entry(self.frame1.sliders, textvariable = variable, justify = "right",
                          takefocus = True, **kwargs)
        if position == 1:
            entry.place(width = 45, relx = 0.35, x = -3, y = 26, anchor = "nw")
        elif position == 2:
            entry.place(width = 45, relx = 0.35, x = -3, y = 71, anchor = "nw")
        elif position == 3:
            entry.place(width = 45, relx = 0.85, x = -3, y = 26, anchor = "nw")
        elif position == 4:
            entry.place(width = 45, relx = 0.85, x = -3, y = 71, anchor = "nw")
        return entry
    
    def pso_widget(self):
        # Initialize widget
        self.init_widget()
        
        # Omega
        self._label("Inertial weight", 1)
        self._scale(0., 1., 0.01, self.w, 1)
        self._entry(self.w, 1)
        
        # C1
        self._label("Cognition parameter", 2)
        self._scale(0., 4., 0.01, self.c1, 2)
        self._entry(self.c1, 2)
        
        # C2
        self._label("Sociability parameter", 3)
        self._scale(0., 4., 0.01, self.c2, 3)
        self._entry(self.c2, 3)
        
    def cpso_widget(self):
        # Initialize widget
        self.pso_widget()
        
        # Gamma
        self._label("Competitivity parameter", 4)
        self._scale(0., 2., 0.01, self.gamma, 4)
        self._entry(self.gamma, 4)
        
    def de_widget(self):
        # Initialize widget
        self.init_widget()
        
        # strategy
        self.strategy_label = ttk.Label(self.frame1.sliders, text = "Strategy")
        self.strategy_option_menu = ttk.OptionMenu(self.frame1.sliders, self.strategy, self.strategy.get(),
                                                   *sorted(self.STRATOPT))
        self.strategy_label.place(relx = 0, x = 0, y = 5, anchor = "nw")
        self.strategy_option_menu.place(relx = 0, x = 70, y = 3, anchor = "nw")
        
        # CR
        self._label("Crossover probability", 2)
        self._scale(0., 1., 0.01, self.CR, 2)
        self._entry(self.CR, 2)

        # F
        self._label("Differential weight", 4)
        self._scale(0., 2., 0.01, self.F, 4)
        self._entry(self.F, 4)
        
    def cmaes_widget(self):
        # Initialize widget
        self.init_widget()
        
        # sigma
        self._label("Step size", 1)
        self._scale(0.01, 10., 0.01, self.sigma, 1)
        self._entry(self.sigma, 1)
        
        # mu_perc
        self._label("Percentage of offsprings", 2)
        self._scale(0.01, 1., 0.01, self.mu_perc, 2)
        self._entry(self.mu_perc, 2)
        
    def hastings_widget(self):
        # Initialize widget
        self.init_widget()
        vcmd1 = self.frame1.pop.register(self._validate_mcmc_stepsize_x1)
        vcmd2 = self.frame1.pop.register(self._validate_mcmc_stepsize_x2)
        
        # stepsize x1
        self._label("Step size X1", 1)
        ss = self._scale(-3., 1., 0.1, self.log_mcmc_stepsize_x1, 1,
                         lambda val: self.mcmc_stepsize_x1.set(round(10.**float(val), 3)))
        ss.set(np.log10(self.mcmc_stepsize_x1.get()))
        self._entry(self.mcmc_stepsize_x1, 1, kwargs = dict(validate = "key",
                                                            validatecommand = (vcmd1, "%P")))
        
        # stepsize x2
        self._label("Step size X2", 2)
        ss = self._scale(-3., 1., 0.1, self.log_mcmc_stepsize_x2, 2,
                         lambda val: self.mcmc_stepsize_x2.set(round(10.**float(val), 3)))
        ss.set(np.log10(self.mcmc_stepsize_x2.get()))
        self._entry(self.mcmc_stepsize_x2, 2, kwargs = dict(validate = "key",
                                                            validatecommand = (vcmd2, "%P")))
        
    def hamiltonian_widget(self):
        # Initialize widget
        self.init_widget()
        vcmd = self.frame1.pop.register(self._validate_hmc_stepsize)
        
        # stepsize
        self._label("Step size", 1)
        ss = self._scale(-3., 1., 0.1, self.log_hmc_stepsize, 1,
                         lambda val: self.hmc_stepsize.set(round(10.**float(val), 3)))
        ss.set(np.log10(self.hmc_stepsize.get()))
        self._entry(self.hmc_stepsize, 1, kwargs = dict(validate = "key",
                                                        validatecommand = (vcmd, "%P")))
        
        # Leap
        self._label("Number of leap frog steps", 2)
        self._scale(1, 100, 1, self.n_leap, 2, lambda val: self.n_leap.set(int(np.floor(float(val)))))
        self._entry(self.n_leap, 2)
    
    def _validate_mcmc_stepsize_x1(self, val):
        try:
            val = float(val)
            if val > 0.:
                self.log_mcmc_stepsize_x1.set(np.log10(val))
        except ValueError:
            pass
        return True
    
    def _validate_mcmc_stepsize_x2(self, val):
        try:
            val = float(val)
            if val > 0.:
                self.log_mcmc_stepsize_x2.set(np.log10(val))
        except ValueError:
            pass
        return True
    
    def _validate_hmc_stepsize(self, val):
        try:
            val = float(val)
            if val > 0.:
                self.log_hmc_stepsize.set(np.log10(val))
        except ValueError:
            pass
        return True
        
    def footer(self):
        # Run button
        run_button = ttk.Button(self.master, text = "Run", command = self.run)

        # Exit button
        exit_button = ttk.Button(self.master, text = "Exit", command = self.close_window)

        # Layout
        run_button.place(relwidth = 0.1, relx = 0.9, rely = 1, x = -5, y = -5, anchor = "se")
        exit_button.place(relwidth = 0.1, relx = 1, rely = 1, x = -5, y = -5, anchor = "se")
        
    def run(self):
        if self.check_variables():
            # To avoid errors when clicking in the window
            if self.first_run:
                self.first_run = False
                
            # To ensure repeatability if needed
            if not self.fix_seed.get():
                self.seed.set(np.random.randint(self.MAX_SEED))
            np.random.seed(self.seed.get())
            
            # Initialize function
            func = "_".join(self.function.get().split()).lower()
            self.bf = BenchmarkFunction(func, n_dim = 2)
            
            # Solve
            solver_name = self.solver_name.get().lower()
            if solver_name in [ "hastings", "hamiltonian" ]:
                if solver_name == "hastings":
                    stepsize = [ self.mcmc_stepsize_x1.get(), self.mcmc_stepsize_x2.get() ]
                else:
                    stepsize = self.hmc_stepsize.get()
                self.solver = MonteCarlo(max_iter = self.max_iter.get(),
                                         constrain = bool(self.constrain.get()),
                                         **self.bf.get())
                self.solver.sample(sampler = solver_name,
                                   stepsize = stepsize,
                                   n_leap = self.n_leap.get())
            elif solver_name in [ "cpso", "pso", "de", "cmaes" ]:
                self.solver = Evolutionary(popsize = self.popsize.get(),
                                           max_iter = self.max_iter.get(),
                                           constrain = bool(self.constrain.get()),
                                           snap = True,
                                           **self.bf.get())
                self.solver.optimize(solver = solver_name,
                                     sync = bool(self.sync.get()),
                                     w = self.w.get(),
                                     c1 = self.c1.get(),
                                     c2 = self.c2.get(),
                                     gamma = self.gamma.get(),
                                     CR = self.CR.get(),
                                     F = self.F.get(),
                                     strategy = self.strategy.get().lower(),
                                     sigma = self.sigma.get(),
                                     mu_perc = self.mu_perc.get())
                
            # Animate
            self.animate(interval = self.interval.get(), yscale = "log")
        
    def animate(self, interval = 100, nx = 101, ny = 101,
                n_levels = 10, yscale = "linear", repeat = True, kwargs = {}):
        # Clear figure
        if self.anim_running:
            self.anim.event_source.stop()
        self.fig.clear()
        
        # Initialize parameters
        models = self.solver.models
        if self.solver._solver in [ "hastings", "hamiltonian" ]:
            func = self._update_monte_carlo
            gfit = self.solver.energy
            frames = models.shape[0]
            linestyle = "--"
            ylabel = "Fitness"
        elif self.solver._solver in [ "cpso", "pso", "de", "cmaes" ]:
            func = self._update_evolutionary
            gfit = self._gfit(self.solver.energy)
            frames = models.shape[-1]
            linestyle = "none"
            ylabel = "Global best fitness"
        max_iter = len(gfit)
        it = np.linspace(1, max_iter, max_iter)
        
        # Initialize axis
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax2 = self.fig.add_subplot(1, 2, 2)
        self.bf.plot(axes = ax1, cont_kws = kwargs)
        self.scatplot, = ax1.plot([], [], linestyle = linestyle, color = "black",
                                  marker = "o",
                                  markersize = 12,
                                  markerfacecolor = "white",
                                  markeredgecolor = "black",
                                  animated = True)
        ax2.plot(it, gfit, linestyle = "-.", linewidth = 1, color = "black")
        self.enerplot, = ax2.plot([], [], linestyle = "-", linewidth = 2,
                                  color = "red")
        ax1.set_xlabel("X1", fontsize = 12)
        ax1.set_ylabel("X2", fontsize = 12)
        ax1.set_xlim(self.bf._lower[0], self.bf._upper[0])
        ax1.set_ylim(self.bf._lower[1], self.bf._upper[1])
        ax2.set_xlim((1, max_iter))
        ax2.set_yscale(yscale)
        ax2.set_ylabel(ylabel, fontsize = 12)
        ax2.grid(True, linestyle = ":")
        self.iter = ax2.text(0.99, 0.99, "", va = "top", ha = "right",
                             fontsize = 10,
                             transform = ax2.transAxes,
                             animated = True)
        
        # Animate
        self.anim_running = True
        self.anim = animation.FuncAnimation(self.fig, func,
                                            fargs = (models, gfit),
                                            frames = frames,
                                            interval = interval,
                                            repeat = repeat,
                                            blit = True)
        self.fig.tight_layout()
    
    def _update_monte_carlo(self, i, models, gfit):
        self.scatplot.set_data(models[:i,0], models[:i,1])
        self.enerplot.set_xdata(np.linspace(1, i+1, i+1))
        self.enerplot.set_ydata(gfit[:i+1])
        self.iter.set_text("Sample %d" % (i+1))
        return self.scatplot, self.enerplot, self.iter,
    
    def _update_evolutionary(self, i, models, gfit):
        self.scatplot.set_data(models[:,0,i], models[:,1,i])
        self.enerplot.set_xdata(np.linspace(1, i+1, i+1))
        self.enerplot.set_ydata(gfit[:i+1])
        self.iter.set_text("Iteration %d" % (i+1))
        return self.scatplot, self.enerplot, self.iter,
    
    def _gfit(self, energy):
        gfit = [ energy[:,0].min() ]
        for i in range(1, energy.shape[1]):
            gfit.append(min(gfit[i-1], energy[:,i].min()))
        return np.array(gfit)
    
    def _onClick(self, event):
        if not self.first_run:
            if self.anim_running:
                self.anim.event_source.stop()
                self.anim_running = False
            else:
                self.anim.event_source.start()
                self.anim_running = True
                
    def export_models(self):
        if self._check_run():
            filename = tkfile.asksaveasfilename(title = "Export models",
                                                filetypes = [ ("Pickle", ".pickle") ],
                                                defaultextension = ".pickle")
            if len(filename) > 0:
                with open(filename, "wb") as f:
                    pickle.dump(self.solver.models, f, protocol = pickle.HIGHEST_PROTOCOL)
    
    def export_fitness(self):
        if self._check_run():
            filename = tkfile.asksaveasfilename(title = "Export fitness",
                                                filetypes = [ ("Pickle", ".pickle") ],
                                                defaultextension = ".pickle")
            if len(filename) > 0:
                with open(filename, "wb") as f:
                    pickle.dump(self.solver.energy, f, protocol = pickle.HIGHEST_PROTOCOL)
    
    def _check_run(self):
        if self.first_run:
            tkmessage.showerror("Error", "No optimization performed yet.")
            return False
        else:
            return True
        
    def close_window(self):
        yes = tkmessage.askyesno("Exit", "Do you really want to quit?")
        if yes:
            self.close()

    def define_variables(self):
        self.solver_name = tk.StringVar(self.master)
        self.function = tk.StringVar(self.master)
        self.popsize = tk.IntVar(self.master)
        self.max_iter = tk.IntVar(self.master)
        self.interval = tk.IntVar(self.master)
        self.mcmc_stepsize_x1 = tk.DoubleVar(self.master)
        self.mcmc_stepsize_x2 = tk.DoubleVar(self.master)
        self.hmc_stepsize = tk.DoubleVar(self.master)
        self.log_mcmc_stepsize_x1 = tk.DoubleVar(self.master)
        self.log_mcmc_stepsize_x2 = tk.DoubleVar(self.master)
        self.log_hmc_stepsize = tk.DoubleVar(self.master)
        self.n_leap = tk.IntVar(self.master)
        self.w = tk.DoubleVar(self.master)
        self.c1 = tk.DoubleVar(self.master)
        self.c2 = tk.DoubleVar(self.master)
        self.gamma = tk.DoubleVar(self.master)
        self.CR = tk.DoubleVar(self.master)
        self.F = tk.DoubleVar(self.master)
        self.strategy = tk.StringVar(self.master)
        self.sigma = tk.DoubleVar(self.master)
        self.mu_perc = tk.DoubleVar(self.master)
        self.seed = tk.IntVar(self.master)
        self.fix_seed = tk.BooleanVar(self.master)
        self.constrain = tk.BooleanVar(self.master)
        self.sync = tk.BooleanVar(self.master)
    
    def trace_variables(self):
        self.solver_name.trace("w", self.callback)
        self.function.trace("w", self.callback)
        self.popsize.trace("w", self.callback)
        self.max_iter.trace("w", self.callback)
        self.interval.trace("w", self.callback)
        self.mcmc_stepsize_x1.trace("w", self.callback)
        self.mcmc_stepsize_x2.trace("w", self.callback)
        self.hmc_stepsize.trace("w", self.callback)
        self.log_mcmc_stepsize_x1.trace("w", self.callback)
        self.log_mcmc_stepsize_x2.trace("w", self.callback)
        self.log_hmc_stepsize.trace("w", self.callback)
        self.n_leap.trace("w", self.callback)
        self.w.trace("w", self.callback)
        self.c1.trace("w", self.callback)
        self.c2.trace("w", self.callback)
        self.gamma.trace("w", self.callback)
        self.CR.trace("w", self.callback)
        self.F.trace("w", self.callback)
        self.strategy.trace("w", self.callback)
        self.sigma.trace("w", self.callback)
        self.mu_perc.trace("w", self.callback)
        self.seed.trace("w", self.callback)
        self.fix_seed.trace("w", self.callback)
        self.constrain.trace("w", self.callback)
        self.sync.trace("w", self.callback)

    def init_variables(self):
        self.solver_name.set("CPSO")
        self.function.set("Rosenbrock")
        self.popsize.set(10)
        self.max_iter.set(200)
        self.interval.set(60)
        self.mcmc_stepsize_x1.set(0.1)
        self.mcmc_stepsize_x2.set(0.1)
        self.hmc_stepsize.set(0.1)
        self.log_mcmc_stepsize_x1.set(np.log10(self.mcmc_stepsize_x1.get()))
        self.log_mcmc_stepsize_x2.set(np.log10(self.mcmc_stepsize_x2.get()))
        self.log_hmc_stepsize.set(np.log10(self.hmc_stepsize.get()))
        self.n_leap.set(10)
        self.w.set(0.72)
        self.c1.set(1.49)
        self.c2.set(1.49)
        self.gamma.set(1.25)
        self.CR.set(0.1)
        self.F.set(0.5)
        self.strategy.set("best2")
        self.sigma.set(1.)
        self.mu_perc.set(0.5)
        self.seed.set(42)
        self.fix_seed.set(False)
        self.constrain.set(True)
        self.sync.set(False)
        
    def check_variables(self):
        # Check popsize
        solver_name = self.solver_name.get().lower()
        if solver_name.upper() in self.EAOPT and self.popsize.get() < self.MIN_POPSIZE[solver_name]:
            tkmessage.showerror("Error", "For %s, population size should be greater than %d." \
                                % (solver_name.upper(), self.MIN_POPSIZE[solver_name]-1))
            return False
        return True
    
    def close(self):
        self.master.quit()
        self.master.destroy()
        
    def callback(self, *args):
        pass
   
    
def main():
    """
    Start StochOPy Viewer window.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    from sys import platform as _platform
    
    root = tk.Tk()
    root.resizable(0, 0)
    StochOGUI(root)
    s = ttk.Style()
    if _platform == "win32":
        s.theme_use("vista")
    elif _platform in [ "linux", "linux2" ]:
        s.theme_use("alt")
    elif _platform == "darwin":
        s.theme_use("aqua")
    root.mainloop()