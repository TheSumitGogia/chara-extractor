import urllib2
from django.utils.encoding import smart_str, smart_unicode
from BeautifulSoup import BeautifulSoup
import re, string, os

class Book:
    def __init__(self, name):
        self.name = name
        self.url = "http://www.sparknotes.com/lit/%s/characters.html" % name
        self.characters = {}

    def getCharactersFromWeb(self):
        try:
            page = urllib2.urlopen(self.url)
        except urllib2.HTTPError as e:
            return
        soup = BeautifulSoup(page.read(), convertEntities=BeautifulSoup.HTML_ENTITIES)
        regex = re.compile('<!--\\n.*DisplayAds.*\\n.*-->')
        for div in soup.findAll("div", {"class" :"content_txt"}):
            name = div.get('id')
            if name == None:
                continue
            # get rid of ads and remove first line
            text = regex.sub("", div.text)
            text = text.split('\n')[1:]
            text = '\n'.join(text)
            text = text.replace('\n', ' ')
            text = text.strip()

            s = text.split('.')
            first_sentence = text.split('.')[0]
            if re.search(name, first_sentence, re.IGNORECASE) == None: 
                if first_sentence.startswith('A ') or first_sentence.startswith('An ') or first_sentence.startswith('The '):
                    first_sentence = first_sentence[0].lower() + first_sentence[1:]
                first_sentence = name + ' is ' + first_sentence
                s[0] = first_sentence
                text = '.'.join(s)
                
            text = smart_str(text)
            self.characters[name] = text

    def writeToFile(self):
        if len(self.characters) == 0:
            return 
        directory = 'books/%s' % self.name
        if not os.path.exists(directory):
            os.makedirs(directory)
        for name in self.characters:
            f = open('%s/%s' % (directory, name), 'w')
            f.write(self.characters[name])

'''
book = Book('elegantuniverse')
book.getCharactersFromWeb()
book.writeToFile()
'''
exclude = set(['elegantuniverse', 'earnest', 'thinair'])
books = set([])
for l in string.ascii_lowercase:
    url = "http://www.sparknotes.com/lit/index_%s.html" % l
    try:
        page = urllib2.urlopen(url)
    except urllib2.HTTPError as e:
        continue 
    html = page.read()
    soup = BeautifulSoup(html)

    for link in soup.findAll('a'):
        url = link.get('href')
        if not url == None and url.startswith("http://www.sparknotes.com/lit/"):
            book = url[30:]
            if len(book) != 0 and book[len(book)-1] == '/':
                book = book[:len(book)-1]
            if book != "" and not book.startswith("index") and not book in books and not book in exclude:
                print 'getting book %s' % book
                books.add(book)
                book = Book(book)
                try:
                    book.getCharactersFromWeb()
                    book.writeToFile()
                except:
                    print "ERROR getting book %s" % book.name
                    exclude.add(book.name)

print exclude
