import numpy
import pyvista

import stochopy

pyvista.set_plot_theme("document")


# Minimize function
fun = stochopy.factory.styblinski_tang
bounds = [[-5.12, 5.12], [-5.12, 5.12]]
options = {"maxiter": 100, "popsize": 20, "constraints": "Shrink", "seed": 0, "return_all": True}

x = stochopy.optimize.pso(fun, bounds, **options)
xall = x.xall
funall = x.funall

# Create surface mesh from objective function topography
xmin, xmax = bounds[0]
ymin, ymax = bounds[1]
x = numpy.linspace(xmin, xmax, 51)
y = numpy.linspace(ymin, ymax, 51)
z = numpy.zeros(1)
x, y, z = numpy.meshgrid(x, y, z)
mesh = pyvista.StructuredGrid(x, y, z).cast_to_unstructured_grid()

funval = numpy.array([fun(point) for point in mesh.points[:, :2]])
scale = 5.12 / funval.max()
mesh.points[:, 2] = funval * scale
mesh["funval"] = funval

# Create point cloud from initial population
points = numpy.column_stack((xall[0], funall[0] * scale))
population = pyvista.PolyData(points)

# Initialize plotter
p = pyvista.Plotter(window_size=(1200, 800), notebook=False)
p.add_mesh(
    mesh,
    scalars="funval",
    stitle="Objective function value",
    scalar_bar_args={
        "height": 0.1,
        "width": 0.5,
        "position_x": 0.01,
        "position_y": 0.01,
        "vertical": False,
        "fmt": "%.1f",
        "title_font_size": 20,
        "font_family": "arial",
        "shadow": True,
    },
    opacity=0.75,
    show_edges=True,
)
p.add_mesh(
    population,
    color="white",
    point_size=15.0,
    render_points_as_spheres=True,
)
generation = p.add_text(
    "Generation 0",
    position="upper_right",
    font_size=12,
    shadow=True,
)
p.show(
    cpos=[
        (6.206969009383443, -14.70514279915864, 14.768395358484344),
        (-0.5534197479501299, -0.2877710118384621, 2.2310160695176586),
        (-0.2570634150903732, 0.5626921760305547, 0.7856818157855469),
    ],
    auto_close=False,
)

# Update population
p.open_movie("sample.mp4", framerate=12)
for i, (x, fun) in enumerate(zip(xall, funall)):
    population.points = numpy.column_stack((x, fun * scale))
    generation.SetText(3, f"Generation {i}")
    p.write_frame()

p.close()

# Convert MP4 to GIF online (e.g., https://convertio.co/)
# GIF produced using imageio are low quality since each frame is limited to 256 colors
