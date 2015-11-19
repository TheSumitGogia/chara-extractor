from bs4 import BeautifulSoup
import os
import re
import urllib2

url = 'http://www.sparknotes.com/lit/%s/characters.html'
sparknotes = os.listdir('books')
gutenberg = open('gutenberg/GUTINDEX.ALL.txt').read().lower()
good_file = open('gutenberg/good.txt', 'w')
bad_file = open('gutenberg/bad.txt', 'w')
ugly_file = open('gutenberg/ugly.txt', 'w')

good, bad, ugly = 0, 0, 0
for book in sparknotes:
  page = urllib2.urlopen(url % book)
  soup = BeautifulSoup(page.read())
  title = soup.findAll('h1', {'class': 'title padding-btm-0'})[0].text.encode('utf-8').lower()
  author = soup.findAll('h2', {'class': 'author'})[0].text.encode('utf-8').lower()
  if len(author) == 0:
    continue
  last_name = author.split(' ')[-1]
  s1 = title in gutenberg
  r = '^(?!audio: ).*%s, by.*%s.*\d+c?(?=\r\n(?!.*\[language: (?!english)))' % (title, last_name)
  s2 = re.search(r, gutenberg, re.M)
  if s1 and s2:
    uid = s2.group(0).split(' ')[-1].strip()
    s = '%s|%s|%s|%s' % (book, title, author, uid)
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
