"""
Example to be used as a "servable" panel with the following command:
>> panel serve example_servable.py --autoreload
"""
import tempfile
import panel as pn

from example import UserTempDirectory, example

dirfunc = UserTempDirectory  # use the user temp location
# dirfunc = tempfile.TemporaryDirectory  # use a TemporaryDirectory

app = example(dirfunc, show_plots=False)

pn.extension()
app.window.servable()
