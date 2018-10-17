import urllib.request
path2="./download_image/"
path="/home/roy/Desktop/Download/ImageNet-downloader-master/n02084071_urls.txt"
days_file = open(path, 'r')

i=1
for a in days_file:
    ID,url=a.split()
    full_file_name =  str(i) + '.jpg'
    full_path="./image/"+full_file_name
    try:
        urllib.request.urlretrieve(url,full_path)
        print ("Download "+str(i)+" image")
        i=i+1

    except Exception as e:
        print(e)
