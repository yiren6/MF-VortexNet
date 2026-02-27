import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_field_distribution(VD, cp, title='Pressure Distribution', min=None, max=None, ax=None, cbar=False, cmap=None, norm= None):
    # Ensure cp is a 1D array
    plt.rcParams.update({
        "font.family": "Helvetica",
        "font.size": 17
    })
    if cp.ndim > 1:
        cp = cp.squeeze(0)

    if min is None:
        vmin = np.min(cp)
    else:
        vmin = min

    if max is None:
        vmax = np.max(cp)
    else:
        vmax = max

    # If no axes are provided, create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Normalize the pressure coefficient values to the range [vmin, vmax]
    if norm == None:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    if cmap == None:
        cmap = plt.get_cmap('viridis')
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Plot the panels and fill with cp data
    for i in range(len(VD.XA1)):
        x = [VD.XA1[i], VD.XB1[i], VD.XB2[i], VD.XA2[i], VD.XA1[i]]
        y = [VD.YA1[i], VD.YB1[i], VD.YB2[i], VD.YA2[i], VD.YA1[i]]

        # Get the color for the current pressure coefficient value
        cp_value = cp[i]
        color = scalar_map.to_rgba(cp_value)

        # Fill the panel with the corresponding cp value
        polygon = plt.Polygon(np.column_stack((x, y)), closed=True, facecolor=color, edgecolor=(0, 0, 0, 0.1))
        ax.add_patch(polygon)

    ax.set_title(title)
    #ax.set_xlabel('X [m]')
    #ax.set_ylabel('Y [m]')
    ax.axis('off')
    ax.set_xlim(np.min(VD.XA1)-0.01, np.max(VD.XB2)+0.01)
    ax.set_ylim(np.min(VD.YA1)-0.01, np.max(VD.YB2)+0.01)
    ax.margins(x=0.002)  # Reduce extra whitespace on X-axis
    # set bounds to tight

    # Return the scalar map for consistent color bar
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)  # Append space for color bar
        cbar_obj = plt.colorbar(scalar_map, cax=cax)
        cbar_obj.set_label(r'$\Delta C_p$')
        return scalar_map
    return scalar_map
