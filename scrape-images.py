"""
Quick script using google_images_download to scape training images from google images.

google_images_download from https://github.com/hardikvasa/google-images-download
"""

from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"Dog Outdoors,Cat Outdoors,Dog Indoors,Cat Indoors","limit":300,"print_urls":True,"color_type":"full-color","thumbnail_only":True,"no_numbering":True,"format":"jpg","output_directory":"data/train", "chromedriver":"/usr/bin/chromedriver"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images