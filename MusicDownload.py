import urllib.request
import urllib.parse

url = 'http://game.ds.qq.com/Com_SongRes/song/dontmatter/dontmatter.mp3'
url = 'http://game.ds.qq.com/Com_SongRes/song/dontmatter/dontmatter_4k_ez.imd'

filename = urllib.parse.unquote(url).split('/')[-1]
print(filename)
filename = 'd:/' + filename


filename, info = urllib.request.urlretrieve(url, filename)
print(filename)
print(info)