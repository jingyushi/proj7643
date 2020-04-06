import click
import glob
import os

def writeFileList(dirNameArr):
    """
    Returns the python list object of the files under a directory name for processing later
    """
    '''
    if isinstance(dirNameArr, basestring): # someone only inputed a single string, so make it a list so that this code works

        dirNameArr = [dirNameArr]
    '''
    dirNameArr = [dirNameArr]
    files_list = [] # list of all files with full path
    for dirName in dirNameArr: 
    # loop through all files in the list of directory names inputted. This is useful for multiple datasets	
        with click.progressbar(os.walk(dirName), label="Parsing files in "+dirName) as bar:
            for dirname, dirnames, filenames in bar:
                for filename in filenames:
                    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.bmp') or filename.endswith('.tiff'):	
                        fileName = glob(os.path.join(dirname, filename)) 
                        files_list += fileName
 
    return files_list,len(files_list)