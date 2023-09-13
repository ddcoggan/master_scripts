def export_legend(legend, filename="legend.pdf"):
	fig = legend.figure
	fig.canvas.draw()
	bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
	fig.savefig(filename, dpi=300, bbox_inches=bbox)

custom_defaults = {'font.size': 20,
                   'lines.linewidth': 3,
                   'lines.markeredgewidth': 2,
                   'lines.markersize': 12,
                   'savefig.dpi': 300,
                   'legend.frameon': False,
                   'ytick.direction': 'in',
                   'ytick.major.width': 1.6,
                   'xtick.direction': 'in',
                   'xtick.major.width': 1.6,
                   'axes.spines.top': False,
                   'axes.spines.right': False,
                   'axes.linewidth': 1.6}

distinct_colours_255 = {
	'red': (230, 25, 75),
	   'green': (60, 180, 75),
	   'yellow': (255, 225, 25),
	   'blue': (0, 130, 200),
	   'orange': (245, 130, 48),
	   'purple': (145, 30, 180),
	   'cyan': (70, 240, 240),
	   'magenta': (240, 50, 230),
	   'lime': (210, 245, 60),
	   'pink': (250, 190, 212),
	   'teal': (0, 128, 128),
	   'lavender': (220, 190, 255),
	   'brown': (170, 110, 40),
	   'beige': (255, 250, 200),
	   'maroon': (128, 0, 0),
	   'mint': (170, 255, 195),
	   'olive': (128, 128, 0),
	   'apricot': (255, 215, 180),
	   'navy': (0, 0, 128),
	   'white': (255, 255, 255),
	   'black': (0, 0, 0)}


distinct_colours = {k: tuple([x / 255. for x in v]) for k, v in distinct_colours_255.items()}