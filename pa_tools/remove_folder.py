"""
function: remove all folder name [need_remove_folder_name]

=> TODO: have to add cli support

"""
import os
import click
import shutil
import logging as log

log.basicConfig(level=log.INFO)


@click.command()
@click.option("--user", help="username for operator cli")
@click.option("--parent_folder", default="")
@click.option("--folder_name", default="", help="Name of need_remove_folder")
def remove_match_folder(user, parent_folder, folder_name):
    """
    remove folder by provided name under the parent_folder
    TODO: add to delete files like [*.xml, *.py, hello* ...]
    """
    if user:
        log.info("hello, {}".format(user))
    if parent_folder == "":
        log.warning("no parent folder provided!")
        return
    for parent, dirnames, filename in os.walk(parent_folder):
        log.info("delete in=> {}".format(parent_folder))
        if folder_name in dirnames:
            need_remove = parent + '/' + folder_name
            log.info("delete folder => {}".format(need_remove))
            shutil.rmtree(need_remove)


remove_match_folder()
