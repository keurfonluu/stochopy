# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib import animation
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog as tkfile
    import tkMessageBox as tkmessage
    from ttk import *
    import tkFont as font
    FileNotFoundError = IOError
else:
    import tkinter as tk
    import tkinter.filedialog as tkfile
    import tkinter.messagebox as tkmessage
    from tkinter.ttk import *
    from tkinter import font
import numpy as np
from benchmark_functions import BenchmarkFunction
from evolutionary_algorithm import Evolutionary
from monte_carlo import MonteCarlo

__all__ = [ "StochOGUI" ]


class StochOGUI():
    
    master = None
    anim_running = False
    
    def __init__(self, master, ncolumn = 2):
        self.nc = ncolumn
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
        
    def menubar(self):
        menubar = tk.Menu(self.master)
        
        # File
        filemenu = tk.Menu(menubar, tearoff = 0)
        filemenu.add_command(label = "Exit", command = self.close_window)
        
        # Help
        helpmenu = tk.Menu(menubar, tearoff = 0)
        helpmenu.add_command(label = "About", command = self.about_window)
        
        # Display menu bar
        menubar.add_cascade(label = "File", menu = filemenu)
        menubar.add_cascade(label = "Help", menu = helpmenu)
        self.master.config(menu = menubar)
        
    def frame1(self):
        self.frame1 = tk.LabelFrame(self.master, text = "Parameters", borderwidth = 2, relief = "groove")
        self.frame1.place(bordermode = "outside", relwidth = 0.99, relheight = 0.2, relx = 0, x = 5, y = 5, anchor = "nw")
        self.frame1.first_run = True
        
        # Function label
        function_label = tk.Label(self.frame1, text = "Function")
        
        # Function menu
        FUNCOPT = ( "Ackley", "Quartic", "Quartic noise", "Rastrigin", "Rosenbrock",
                    "Sphere", "Styblinski-Tang" )
        function_option_menu = tk.OptionMenu(self.frame1, self.function, *FUNCOPT)
        
        # Iteration label
        max_iter_label = tk.Label(self.frame1, text = "Maximum number of iterations")
        
        # Iteration spinbox
        max_iter_spinbox = tk.Spinbox(self.frame1, from_ = 1, to_ = 999999,
                                      increment = 1, textvariable = self.max_iter,
                                      justify = "right")
        
        # FPS label
        fps_label = tk.Label(self.frame1, text = "Number of frame per second")
        
        # FPS spinbox
        fps_spinbox = tk.Spinbox(self.frame1, from_ = 1, to_ = 1000,
                                 increment = 1, textvariable = self.interval,
                                 justify = "right")
        
        # Solver label
        solver_label = tk.Label(self.frame1, text = "Solver")
        
        # Solver option menu
        SOLVOPT = ( "Hastings", "Hamiltonian", "CPSO", "PSO", "DE", "CMAES" )
        solver_option_menu = tk.OptionMenu(self.frame1, self.solver, *SOLVOPT,
                                           command = self.select_widget)
        
        # Popsize label
        popsize_label = tk.Label(self.frame1, text = "Population size")
        
        # Popsize spinbox
        popsize_spinbox = tk.Spinbox(self.frame1, from_ = 1, to_ = 1000,
                                     increment = 1, textvariable = self.popsize,
                                     justify = "right")
        
        # Layout
        function_label.place(relx = 0., x = 5, y = 5, anchor = "nw")
        function_option_menu.place(relx = 0., x = 60, anchor = "nw")
        max_iter_label.place(relx = 0., x = 5, y = 50, anchor = "nw")
        max_iter_spinbox.place(width = 60, relx = 0., x = 180, y = 50, anchor = "nw")
        fps_label.place(relx = 0., x = 5, y = 75, anchor = "nw")
        fps_spinbox.place(width = 60, relx = 0., x = 180, y = 75, anchor = "nw")
        solver_label.place(relx = 0.3, x = 0, y = 5, anchor = "nw")
        solver_option_menu.place(relx = 0.3, x = 50, anchor = "nw")
        popsize_label.place(relx = 0.3, x = 0, y = 50, anchor = "nw")
        popsize_spinbox.place(width = 60, relx = 0.3, x = 100, y = 50, anchor = "nw")
        
        # Call widget
        self.select_widget("CPSO")
    
    def frame2(self):
        self.frame2 = tk.Frame(self.master, borderwidth = 2, relief = "groove")
        self.frame2.place(bordermode = "outside", relwidth = 0.99, relheight = 0.73, relx = 0, rely = 0.21, x = 5, y = 5, anchor = "nw")
        self.frame2_in = tk.Frame(self.frame2, borderwidth = 0)
        self.frame2_in.place(relwidth = 1, relheight = 1, relx = 0, anchor = "nw")
        self.fig = Figure(figsize = (13, 6), facecolor = "white")
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.frame2_in)
        self.fig.canvas.mpl_connect("button_press_event", self._onClick)
        self.canvas.get_tk_widget().pack()
        
    def init_widget(self):
        if not self.frame1.first_run:
            self.frame1_in.forget()
        else:
            self.frame1.first_run = False
        self.frame1_in = tk.Frame(self.frame1, borderwidth = 0)
        self.frame1_in.place(relwidth = 0.5, relheight = 1., relx = 0.5, anchor = "nw")
        
    def select_widget(self, solver):
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
        
    def pso_widget(self):
        # Initialize widget
        self.init_widget()
        
        # Omega label
        omega_label = tk.Label(self.frame1_in, text = "Inertial weight")

        # Omega scale
        omega_scale = tk.Scale(self.frame1_in, from_ = 0., to_ = 1., resolution = 0.01,
                               variable = self.w, showvalue = 0,
                               orient = "horizontal", borderwidth = 1,
                               width = 15, sliderlength = 20, sliderrelief = "ridge")

        # Omega entry
        omega_entry = tk.Entry(self.frame1_in, textvariable = self.w, justify = "right")
        
        # C1 label
        c1_label = tk.Label(self.frame1_in, text = "Cognition parameter")
        
        # C1 scale
        c1_scale = tk.Scale(self.frame1_in, from_ = 0., to_ = 4., resolution = 0.01,
                            variable = self.c1, showvalue = 0,
                            orient = "horizontal", borderwidth = 1,
                            width = 15, sliderlength = 20, sliderrelief = "ridge")
        
        # C1 entry
        c1_entry = tk.Entry(self.frame1_in, textvariable = self.c1, justify = "right")
        
        # C2 label
        c2_label = tk.Label(self.frame1_in, text = "Sociability parameter")
        
        # C2 scale
        c2_scale = tk.Scale(self.frame1_in, from_ = 0., to_ = 4., resolution = 0.01,
                            variable = self.c2, showvalue = 0,
                            orient = "horizontal", borderwidth = 1,
                            width = 15, sliderlength = 20, sliderrelief = "ridge")
        
        # C2 entry
        c2_entry = tk.Entry(self.frame1_in, textvariable = self.c2, justify = "right")
        
        # Layout
        omega_label.place(relx = 0, x = 0, y = 5, anchor = "nw")
        omega_scale.place(relwidth = 0.35, relx = 0, x = 0, y = 25, anchor = "nw")
        omega_entry.place(relwidth = 0.1, relx = 0.35, x = -3, y = 26, anchor = "nw")
        c1_label.place(relx = 0, x = 0, y = 50, anchor = "nw")
        c1_scale.place(relwidth = 0.35, relx = 0, x = 0, y = 70, anchor = "nw")
        c1_entry.place(relwidth = 0.1, relx = 0.35, x = -3, y = 71, anchor = "nw")
        c2_label.place(relx = 0.5, x = 0, y = 5, anchor = "nw")
        c2_scale.place(relwidth = 0.35, relx = 0.5, x = 0, y = 25, anchor = "nw")
        c2_entry.place(relwidth = 0.1, relx = 0.85, x = -3, y = 26, anchor = "nw")
        
    def cpso_widget(self):
        # Initialize widget
        self.pso_widget()
        
        # gamma label
        gamma_label = tk.Label(self.frame1_in, text = "Competitivity parameter")
        
        # gamma scale
        gamma_scale = tk.Scale(self.frame1_in, from_ = 0., to_ = 2., resolution = 0.01,
                               variable = self.gamma, showvalue = 0,
                               orient = "horizontal", borderwidth = 1,
                               width = 15, sliderlength = 20, sliderrelief = "ridge")
        
        # gamma entry
        gamma_entry = tk.Entry(self.frame1_in, textvariable = self.gamma, justify = "right")
        
        # Layout
        gamma_label.place(relx = 0.5, x = 0, y = 50, anchor = "nw")
        gamma_scale.place(relwidth = 0.35, relx = 0.5, x = 0, y = 70, anchor = "nw")
        gamma_entry.place(relwidth = 0.1, relx = 0.85, x = -3, y = 71, anchor = "nw")
        
    def de_widget(self):
        # Initialize widget
        self.init_widget()
        
        # CR label
        CR_label = tk.Label(self.frame1_in, text = "Crossover probability")

        # CR scale
        CR_scale = tk.Scale(self.frame1_in, from_ = 0., to_ = 1., resolution = 0.01,
                            variable = self.CR, showvalue = 0,
                            orient = "horizontal", borderwidth = 1,
                            width = 15, sliderlength = 20, sliderrelief = "ridge")

        # CR entry
        CR_entry = tk.Entry(self.frame1_in, textvariable = self.CR, justify = "right")

        # F label
        F_label = tk.Label(self.frame1_in, text = "Differential weight")

        # F scale
        F_scale = tk.Scale(self.frame1_in, from_ = 0., to_ = 2., resolution = 0.01,
                           variable = self.F, showvalue = 0,
                           orient = "horizontal", borderwidth = 1,
                           width = 15, sliderlength = 20, sliderrelief = "ridge")

        # F entry
        F_entry = tk.Entry(self.frame1_in, textvariable = self.F, justify = "right")

        # Layout
        CR_label.place(relx = 0, x = 0, y = 5, anchor = "nw")
        CR_scale.place(relwidth = 0.35, relx = 0, x = 0, y = 25, anchor = "nw")
        CR_entry.place(relwidth = 0.1, relx = 0.35, x = -3, y = 26, anchor = "nw")
        F_label.place(relx = 0, x = 0, y = 50, anchor = "nw")
        F_scale.place(relwidth = 0.35, relx = 0, x = 0, y = 70, anchor = "nw")
        F_entry.place(relwidth = 0.1, relx = 0.35, x = -3, y = 71, anchor = "nw")
        
    def cmaes_widget(self):
        # Initialize widget
        self.init_widget()
        
        # mu label
        mu_label = tk.Label(self.frame1_in, text = "Percentage of offspring")

        # mu scale
        mu_scale = tk.Scale(self.frame1_in, from_ = 0., to_ = 1., resolution = 0.01,
                            variable = self.mu_perc, showvalue = 0,
                            orient = "horizontal", borderwidth = 1,
                            width = 15, sliderlength = 20, sliderrelief = "ridge")

        # mu entry
        mu_entry = tk.Entry(self.frame1_in, textvariable = self.mu_perc, justify = "right")

        # Layout
        mu_label.place(relx = 0, x = 0, y = 5, anchor = "nw")
        mu_scale.place(relwidth = 0.35, relx = 0, x = 0, y = 25, anchor = "nw")
        mu_entry.place(relwidth = 0.1, relx = 0.35, x = -3, y = 26, anchor = "nw")
        
    def hastings_widget(self):
        # Initialize widget
        self.init_widget()
        
        # stepsize label
        stepsize_label = tk.Label(self.frame1_in, text = "Step size")

        # stepsize scale
        stepsize_scale = tk.Scale(self.frame1_in, from_ = -10., to_ = 2., resolution = 0.1,
                                  command = self._log_scale, showvalue = 0,
                                  orient = "horizontal", borderwidth = 1,
                                  width = 15, sliderlength = 20, sliderrelief = "ridge")

        # stepsize entry
        stepsize_entry = tk.Entry(self.frame1_in, textvariable = self.stepsize, justify = "right")

        # Layout
        stepsize_label.place(relx = 0, x = 0, y = 5, anchor = "nw")
        stepsize_scale.place(relwidth = 0.35, relx = 0, x = 0, y = 25, anchor = "nw")
        stepsize_entry.place(relwidth = 0.1, relx = 0.35, x = -3, y = 26, anchor = "nw")
        
    def hamiltonian_widget(self):
        # Initialize widget
        self.hastings_widget()
        
        # Leap label
        leap_label = tk.Label(self.frame1_in, text = "Number of leap frog steps")

        # Leap scale
        leap_scale = tk.Scale(self.frame1_in, from_ = 1, to_ = 100, resolution = 1,
                           variable = self.n_leap, showvalue = 0,
                           orient = "horizontal", borderwidth = 1,
                           width = 15, sliderlength = 20, sliderrelief = "ridge")

        # Leap entry
        leap_entry = tk.Entry(self.frame1_in, textvariable = self.n_leap, justify = "right")

        # Layout
        leap_label.place(relx = 0, x = 0, y = 50, anchor = "nw")
        leap_scale.place(relwidth = 0.35, relx = 0, x = 0, y = 70, anchor = "nw")
        leap_entry.place(relwidth = 0.1, relx = 0.35, x = -3, y = 71, anchor = "nw")
        
    def _log_scale(self, val):
        self.stepsize.set(10.**float(val))
    
    def footer(self):
        # Run button
        run_button = tk.Button(self.master, text = "Run", command = self.run)

        # Exit button
        exit_button = tk.Button(self.master, text = "Exit", command = self.close_window)

        # Layout
        run_button.place(relwidth = 0.1, relx = 0.9, rely = 1, x = -5, y = -5, anchor = "se")
        exit_button.place(relwidth = 0.1, relx = 1, rely = 1, x = -5, y = -5, anchor = "se")
        
    def about_window(self):
        about = "StochOPy Viewer 1.0" + "\n" \
                + "Created by Keurfon Luu"
        tkmessage.showinfo("About", about)
        
    def run(self):
        func = "_".join(self.function.get().split()).lower()
        solver = self.solver.get().lower()
        self.bf = BenchmarkFunction(func, n_dim = 2)
        if solver in [ "hastings", "hamiltonian" ]:
            op = MonteCarlo(**self.bf.get(), max_iter = self.max_iter.get())
            op.sample(sampler = solver,
                      stepsize = self.stepsize.get(),
                      n_leap = self.n_leap.get())
        elif solver in [ "cpso", "pso", "de", "cmaes" ]:
            op = Evolutionary(**self.bf.get(), popsize = self.popsize.get(),
                              max_iter = self.max_iter.get())
            op.optimize(solver = solver, snap = True,
                        w = self.w.get(),
                        c1 = self.c1.get(),
                        c2 = self.c2.get(),
                        gamma = self.gamma.get(),
                        CR = self.CR.get(),
                        F = self.F.get(),
                        mu_perc = self.mu_perc.get())
        self.animate(op, interval = self.interval.get(), yscale = "log")
        
    def animate(self, solver, interval = 100, nx = 101, ny = 101,
                n_levels = 10, yscale = "linear", repeat = True, kwargs = {}):
        if self.anim_running:
            self.anim.event_source.stop()
        self.fig.clear()
        models = solver.models
        if solver._solver in [ "hastings", "hamiltonian" ]:
            func = self._update_monte_carlo
            gfit = solver.energy
            linestyle = "--"
            xlabel = "Sample number"
            ylabel = "Fitness"
        elif solver._solver in [ "cpso", "pso", "de", "cmaes" ]:
            func = self._update_evolutionary
            gfit = self._gfit(solver.energy)
            linestyle = "none"
            xlabel = "Iteration number"
            ylabel = "Global best fitness"
        else:
            raise ValueError("solver should either be an instance MonteCarlo or Evolutionary")
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax2 = self.fig.add_subplot(1, 2, 2)
        self.bf.plot(axes = ax1, kwargs = kwargs)
        self.scatplot, = ax1.plot([], [], linestyle = linestyle, color = "black",
                                  marker = "o",
                                  markersize = 12,
                                  markerfacecolor = "white",
                                  markeredgecolor = "black")
        ax2.plot(gfit, linestyle = "-.", linewidth = 1, color = "black")
        self.enerplot, = ax2.plot([], [], linestyle = "-", linewidth = 2,
                                  color = "red")
        ax1.set_xlim(self.bf._lower[0], self.bf._upper[0])
        ax1.set_ylim(self.bf._lower[1], self.bf._upper[1])
        ax2.set_xlim((0, len(gfit)))
        ax2.set_yscale(yscale)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.grid(True)
        
        self.anim_running = True
        self.anim = animation.FuncAnimation(self.fig, func,
                                            fargs = (models, gfit),
                                            frames = models.shape[-1],
                                            interval = interval,
                                            repeat = repeat,
                                            blit = True)
        self.fig.tight_layout()
    
    def _update_monte_carlo(self, i, models, gfit):
        self.scatplot.set_data(models[0,:i], models[1,:i])
        self.enerplot.set_xdata(np.arange(i+1))
        self.enerplot.set_ydata(gfit[:i+1])
        return self.scatplot, self.enerplot,
    
    def _update_evolutionary(self, i, models, gfit):
        self.scatplot.set_data(models[0,:,i], models[1,:,i])
        self.enerplot.set_xdata(np.arange(i+1))
        self.enerplot.set_ydata(gfit[:i+1])
        return self.scatplot, self.enerplot,
    
    def _gfit(self, energy):
        gfit = [ energy[:,0].min() ]
        for i in range(1, energy.shape[1]):
            gfit.append(min(gfit[i-1], energy[:,i].min()))
        return np.array(gfit)
    
    def _onClick(self, event):
        if self.anim_running:
            self.anim.event_source.stop()
            self.anim_running = False
        else:
            self.anim.event_source.start()
            self.anim_running = True
        
    def close_window(self):
        yes = tkmessage.askyesno("Exit", "Do you really want to quit?")
        if yes:
            self.close()

    def define_variables(self):
        self.solver = tk.StringVar(self.master)
        self.function = tk.StringVar(self.master)
        self.popsize = tk.IntVar(self.master)
        self.max_iter = tk.IntVar(self.master)
        self.interval = tk.IntVar(self.master)
        self.stepsize = tk.DoubleVar(self.master)
        self.n_leap = tk.IntVar(self.master)
        self.w = tk.DoubleVar(self.master)
        self.c1 = tk.DoubleVar(self.master)
        self.c2 = tk.DoubleVar(self.master)
        self.gamma = tk.DoubleVar(self.master)
        self.CR = tk.DoubleVar(self.master)
        self.F = tk.DoubleVar(self.master)
        self.sigma = tk.DoubleVar(self.master)
        self.mu_perc = tk.DoubleVar(self.master)
    
    def trace_variables(self):
        self.solver.trace("w", self.callback)
        self.function.trace("w", self.callback)
        self.popsize.trace("w", self.callback)
        self.max_iter.trace("w", self.callback)
        self.interval.trace("w", self.callback)
        self.stepsize.trace("w", self.callback)
        self.n_leap.trace("w", self.callback)
        self.w.trace("w", self.callback)
        self.c1.trace("w", self.callback)
        self.c2.trace("w", self.callback)
        self.gamma.trace("w", self.callback)
        self.CR.trace("w", self.callback)
        self.F.trace("w", self.callback)
        self.sigma.trace("w", self.callback)
        self.mu_perc.trace("w", self.callback)

    def init_variables(self):
        self.solver.set("CPSO")
        self.function.set("Rosenbrock")
        self.popsize.set(10)
        self.max_iter.set(100)
        self.interval.set(60)
        self.stepsize.set(0.1)
        self.n_leap.set(10)
        self.w.set(0.72)
        self.c1.set(1.49)
        self.c2.set(1.49)
        self.gamma.set(1.25)
        self.CR.set(0.5)
        self.F.set(1.)
        self.sigma.set(1.)
        self.mu_perc.set(0.5)
    
    def close(self):
        self.master.quit()
        self.master.destroy()
        
    def callback(self, *args):
        pass
    

if __name__ == "__main__":
    root = tk.Tk()
    #root.attributes('-zoomed', True)
    gui = StochOGUI(root)
    root.mainloop()