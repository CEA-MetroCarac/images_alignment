"""
Utilities functions or Classes
"""
import os
import re
import glob
from itertools import groupby, count
from tkinter import (Frame, Button, Listbox, Text, Label, Entry,
                     W, E, END, Y, LEFT, RIGHT)
from tkinter import filedialog as fd
from tkinter.ttk import Scrollbar
from tkinter.font import Font


def add(obj, row, col, sticky='', padx=5, pady=3, rspan=1, cspan=1, **kwargs):
    """ Add tkinter object at the (row, col)-position of a grid """
    obj.grid(row=row, column=col, sticky=sticky, padx=padx, pady=pady,
             rowspan=rspan, columnspan=cspan, **kwargs)


def add_entry(frame, row, label, val, width=5, bind_fun=None):
    """ Add 2 columns : 'label' and 'val' associated to a binding function
        'bind_fun' at 'row' (int) position in a 'frame' (Tk.Frame) """
    add(Label(frame, text=label), row, 0, E)
    entry = Entry(frame, textvariable=val, width=width)
    add(entry, row, 1, W)
    if bind_fun is not None:
        entry.bind('<Return>', lambda event: bind_fun())


def hsorted(list_):
    """ Sort the given list in the way that humans expect """
    list_ = [str(x) for x in list_]
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(list_, key=alphanum_key)


class Terminal(Text):
    """ Class inherited from Tkinter.Text with a write() function """

    def __init__(self, root):
        font = Font(family="Helvetica", size=10)
        super().__init__(root, height=8, width=25, font=font)

    def write(self, value):
        """ Write 'value' in the related Tkinter.Text object """
        self.insert(END, value)
        self.see(END)
        self.update_idletasks()


class FilesSelector:
    """
    Class dedicated to the files selection

    Attributes
    ----------
    lbox: list of Listbox object
        Listbox associated to the selected files
    fnames: list of str
        List of filenames to work with

    Parameters
    ----------
    root: Tk.widget
        The main window associated to the FileSelector
    lbox_size: list of 2 ints, optional
        Size (width, height) of the Listbox 'lbox'. Default value is [30, 15]
    """

    def __init__(self, root, lbox_size=None):

        lbox_size = lbox_size or [30, 15]

        self.fnames = []

        # create buttons and listbox

        Button(root, text="Select Files",
               command=self.select_files). \
            grid(column=0, row=0, padx=0, pady=5)
        Button(root, text="Select Dir.",
               command=self.select_dir). \
            grid(column=1, row=0, padx=0, pady=5)
        Button(root, text="Remove",
               command=self.remove). \
            grid(column=2, row=0, padx=0, pady=5)
        Button(root, text="Remove all",
               command=self.remove_all). \
            grid(column=3, row=0, padx=0, pady=5)

        Button(root, text="▲", command=lambda: self.move('up')). \
            grid(column=1, row=1, pady=0, sticky=E)
        Button(root, text="▼", command=lambda: self.move('down')). \
            grid(column=2, row=1, pady=0, sticky=W)

        lbox_frame = Frame(root)
        lbox_frame.grid(column=0, row=2, padx=5, pady=0, columnspan=4)
        sbar_v = Scrollbar(lbox_frame)
        # sbar_h = Scrollbar(lbox_frame, orient=HORIZONTAL)
        self.lbox = Listbox(lbox_frame,
                            width=lbox_size[0],
                            height=lbox_size[1],
                            # selectmode=EXTENDED,
                            activestyle="underline",
                            exportselection=False,
                            # xscrollcommand=sbar_h.set,
                            yscrollcommand=sbar_v.set)

        sbar_v.config(command=self.lbox.yview)
        # sbar_h.config(command=self.lbox.xview)

        # sbar_h.pack(side=BOTTOM, fill=X)
        self.lbox.pack(side=LEFT, fill=Y)
        sbar_v.pack(side=RIGHT, fill=Y)

    def move(self, key):
        """ Move cursor selection according to key value (up or down) """
        increment = {'up': -1, 'down': 1, 'none': 0}
        indices = self.lbox.curselection()
        if not indices:
            return
        ind = min(indices)  # working with several indices has no sense
        if len(indices) > 1:
            key = 'none'
        elif (key == 'up' and ind == 0) or \
                (key == 'down' and ind == len(self.fnames) - 1):
            return
        ind += increment[key]
        self.select_item(ind)

        self.lbox.event_generate('<<ListboxSelect>>')

    def add_items(self, fnames=None, ind_start=None):
        """ Add items from a 'fnames' list """
        if fnames is None:
            return

        ind_start = ind_start or len(self.fnames)

        for fname in hsorted(fnames):
            self.lbox.insert(END, os.path.basename(fname))
            self.fnames.append(fname)

        # select the first new item
        self.select_item(ind_start)

    def select_files(self, fnames=None):
        """ Add items from selected files """

        if fnames is None:
            fnames = fd.askopenfilenames(title='Select file(s)')
        self.add_items(fnames=fnames)

        self.lbox.event_generate('<<ListboxAdd>>')

    def select_dir(self, dirname=None):
        """ Add items from a directory """
        if dirname is None:
            dirname = fd.askdirectory(title='Select directory')

        ind_start = len(self.fnames)
        fnames = glob.glob(os.path.join(dirname, '*.tif'))
        self.add_items(fnames, ind_start=ind_start)

        self.lbox.event_generate('<<ListboxAdd>>')

    def remove(self):
        """ Remove selected items """
        selected_items = self.lbox.curselection()

        # deleting one by one item is too long when working with a big selection
        groups = groupby(selected_items, key=lambda n, c=count(): n - next(c))
        groups = [list(g) for _, g in groups]
        for group in groups:
            self.lbox.delete(group[0], group[-1])

        for selected_item in reversed(selected_items):
            self.fnames.pop(selected_item)

        # reselect the first item
        self.select_item(0)

        self.lbox.event_generate('<<ListboxRemove>>')

    def remove_all(self):
        """ Remove all items """
        self.lbox.delete(0, END)
        self.fnames = []

        self.lbox.event_generate('<<ListboxRemoveAll>>')

    def select_item(self, index, selection_clear=True):
        """ Select item in the listbox """
        if selection_clear:
            self.lbox.selection_clear(0, END)
        self.lbox.selection_set(index)
        self.lbox.activate(index)
        self.lbox.selection_anchor(index)
        self.lbox.see(index)
