import urllib.parse
import urllib.request, os
import requests
import LatLongData

myloc = r"./Location Images" #Replace with correct location
key = "&key=" + "AIzaSyA8RcVdojD1-GWoNTOPSES8KPGC1swbp2w" #got banned after ~100 requests with no key or after 25,000 requests without signature
secret = "F_KY54alKrs5mlVbR_OVDj9BA-E="

def getLatLongImage(Loc, SaveLoc, index):
  base = "https://maps.googleapis.com/maps/api/streetview?size=640x640&radius=100&location="

  MyUrl = base + urllib.parse.quote_plus(Loc) + key #added url encoding
  #print(MyUrl)
  fi = str(index) + ".jpg"
  urllib.request.urlretrieve(MyUrl, os.path.join(SaveLoc, fi))

def checkLatLongImage(Loc):
  metadata_base = "https://maps.googleapis.com/maps/api/streetview/metadata?&location="
  metadata_url = metadata_base + urllib.parse.quote_plus(Loc) + key #added url encoding
  response = requests.post(metadata_url)
  response_data = response.json()
  return response_data["status"]

#List of coordinates for testing
tests = {1: "37.79101665,-122.3991486", #01ST ST \ BUSH ST \ MARKET ST
         2: "37.78771761,-122.3950078", #01ST ST \ CLEMENTINA ST
         3: "37.81437339484346,-122.3583278940174", #No image within 50m (radius of 60m finds an image)
         4: "37.79005267,-122.3979386"} #01ST ST \ END
         #Note: Use something like

#db = tests

#Actual list of coordinates
db = LatLongData.getCoords()

for loc in db:
  if checkLatLongImage(db[loc]) == "OK":
    getLatLongImage(Loc=db[loc], SaveLoc=myloc, index = loc)
  