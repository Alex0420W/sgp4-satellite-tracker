"""Ground-track visualisation: static (cartopy) and interactive (plotly).

The pure-numerical helpers in :mod:`sat_tracker.visualization.common` have no
plotting-library dependencies and are unit-tested in isolation. The cartopy
and plotly renderers each defer-import their respective heavy dependencies so
a user without ``[viz]`` extras installed can still import this package
without errors — the failure surfaces only when they actually try to render.
"""
