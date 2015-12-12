#!/usr/bin/env bash
#
# Tokenizer taking new lines into account
echo java -mx3g -cp \"$CORE_NLP/*\" edu.stanford.nlp.process.PTBTokenizer -options tokenizeNLs=true $* 
indir="$1/*"
outdir="$2"
mkdir -p $outdir
for f in $indir
do
    echo "Tokenizing $f ..."
    java -mx3g -cp "$CORE_NLP/*" edu.stanford.nlp.process.PTBTokenizer -options tokenizeNLs=true "$f" > "$outdir/$(basename $f)"
done
