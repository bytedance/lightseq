
N_LINE=1000000
trainpref=train
validpref=valid
testpref=test
workers=60
extra=''

until [[ -z "$1" ]]
do
    case $1 in
        --source-lang)
            shift; SRC=$1;
            shift;;
        --target-lang)
            shift; TGT=$1;
            shift;;
        --trainpref)
            shift; trainpref=$1;
            shift;;
        --validpref)
            shift; validpref=$1;
            shift;;
        --testpref)
            shift; testpref=$1;
            shift;;
        --destdir)
            shift; DESTDIR=$1;
            shift;;
        --n-line-per-file)
            shift; N_LINE=$1;
            shift;;
        --text-dir)
            shift; TEXT=$1;
            shift;;
        *)
            extra=${extra}\ ${1}
            shift;;
    esac
done


if [[ ! -d $DESTDIR ]]; then
    mkdir -p $DESTDIR
fi
rm -f $DESTDIR/*

DESTDIR=$(readlink -f $DESTDIR)
TEXT=$(readlink -f $TEXT)
TMP=/tmp/build_split_k_databin

if [[ ! -d $TMP ]]; then
    mkdir -p $TMP
else
    echo "delete $TMP"
    rm -rf $TMP
    mkdir -p $TMP
fi
cd $TMP

echo "split big file into lots of small files......"
split -l $N_LINE $TEXT/$trainpref.$SRC src -d -a 6
for i in `ls src*`; do
    new_str=$(echo -e $i | sed -r 's/src0*([0-9])/\1/')  
    mv $i train$new_str.$SRC;
done
mv train0.$SRC train.$SRC

split -l $N_LINE $TEXT/$trainpref.$TGT tgt -d -a 6
for i in `ls tgt*`; do
    new_str=$(echo -e $i | sed -r 's/tgt0*([0-9])/\1/')  
    mv $i train$new_str.$TGT;
done
mv train0.$TGT train.$TGT

# generate dictionary files
fairseq-preprocess \
    --source-lang $SRC --target-lang $TGT \
    --trainpref $TEXT/$trainpref --validpref $TEXT/$validpref --testpref $TEXT/$testpref \
    --destdir $TMP --thresholdtgt 0 --thresholdsrc 0 \
    --workers 60 $extra


sd=$TMP/dict.$SRC.txt
td=$TMP/dict.$TGT.txt
# binarize corpus
for i in `ls $TMP/train*.$SRC`; do
    new_str=$(echo -e $i | cut -d"." -f1 )
    type_name=$(echo $new_str | rev | cut -d"/" -f1 | rev)
    fairseq-preprocess \
        --source-lang $SRC --target-lang $TGT \
        --srcdict $sd --tgtdict $td \
        --trainpref $new_str --validpref $TEXT/$validpref --testpref $TEXT/$testpref \
        --destdir databins/$type_name --thresholdtgt 0 --thresholdsrc 0 \
        --workers 60 $extra
done

# reduce all files in destdir
cp $TMP/databins/train/* $DESTDIR/
cp -f $TMP/preprocess.log $DESTDIR/

for i in `ls $TMP/databins/`; do
    for j in `ls $TMP/databins/$i/train*`; do
        type_name=$(echo $j | rev | cut -d"/" -f1 | rev)
        new_name=$(echo $type_name | sed "s/train/$i/")
        cp $j $DESTDIR/$new_name
    done
done

