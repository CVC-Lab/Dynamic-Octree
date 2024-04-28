import os, shutil

def cleanDirectory(dir):
    r"""
    Cleans and prepares a directory for generateTrajectories and
    generateAtoms

    args:
        dir (str): Directory to clean
    """

    # Data directory doesn't exist
    if not (os.path.exists(dir)):
        os.makedirs(dir)
    # Data directory is already populated
    if len(os.listdir(dir)) != 0:
        shutil.rmtree(dir)
        os.makedirs(dir)
