from bs4 import BeautifulSoup
import os
import re
import urllib2

sparknotes_url = 'http://www.sparknotes.com/lit/%s/characters.html'
google_url = 'http://www.google.com/search?q=%s %s'
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.86 Safari/537.36'
headers = {'User-Agent': user_agent}
sparknotes = os.listdir('books')
gutenberg = open('gutenberg/GUTINDEX.ALL.txt').read().lower()
good_file = open('gutenberg/good.txt', 'w')
bad_file = open('gutenberg/bad.txt', 'w')
ugly_file = open('gutenberg/ugly.txt', 'w')
blacklist_path = 'gutenberg/blacklist.txt'
blacklist = set([])

with open(blacklist_path) as f:
  for book in f:
    blacklist.add(book.strip())

good, bad, ugly = 0, 0, 0
for book in sparknotes:
  if book in blacklist:
    continue
  # Sparknotes fail check
  try:
    page = urllib2.urlopen(sparknotes_url % book)
  except urllib2.HTTPError:
    blacklist.add(book)
    continue
  # Get title and author from sparknotes
  soup = BeautifulSoup(page.read())
  title = soup.findAll('h1', {'class': 'title padding-btm-0'})[0].text.encode('utf-8').lower()
  author = soup.findAll('h2', {'class': 'author'})[0].text.encode('utf-8').lower()
  if len(author) == 0:
    blacklist.add(book)
    continue
  last_name = author.split(' ')[-1]

  # Get type of literary work from google
  url = (google_url % (title, author if author != 'anonymous' else '')).strip().replace(' ', '%20')
  request = urllib2.Request(url, None, headers)
  page = urllib2.urlopen(request)
  soup = BeautifulSoup(page.read())
  tag = soup.findAll('div', {'class': '_gdf'})
  if len(tag) == 0:
    blacklist.add(book)
    continue
  desc = soup.findAll('div', {'class': '_gdf'})[0].text.lower()
  if 'play' in desc or 'poem' in desc:
    blacklist.add(book)
    continue

  # Author in gutenberg index
  s1 = title in gutenberg
  # Author and title in gutenberg index in same entry
  r = '^(?!audio: ).*%s, by.*%s.*\d+c?\r\n?(?:.+\r\n?)*' % (title, last_name)
  s2 = re.findall(r, gutenberg, re.M)
  # Filter our non english entries
  gids = []
  for entry in s2:
    spl = entry.splitlines()
    if not any(re.search('language: (?!english)', s) for s in spl[1:]):
      gid = int(spl[0].split()[-1].strip().replace('c', ''))
      gids.append(gid)

  if len(gids) > 0:
    gid = min(gids)
    s = '%s|%s|%s|%s' % (book, title, author, gid)
    print s
    good_file.write('%s\n' % s)
    good += 1
  elif s1 and not s2:
    s = '%s|%s|%s' % (book, title, author)
    print s
    ugly_file.write('%s\n' % s)
    ugly += 1
  else:
    s = '%s|%s|%s' % (book, title, author)
    print s
    bad_file.write('%s\n' % s)
    bad += 1

print good, bad, ugly

with open(blacklist_path, 'w') as f:
  for book in blacklist:
    f.write('%s\n' % book)
