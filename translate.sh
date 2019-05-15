# Перевод предложений из файла, указанного первым параметром с сохранением перевода в файл, указанный вторым параметром.
# Пример: ./translate.sh /home/ivanov/data/newstest.src.txt /home/ivanov/data/newstest.tgt.txt

############## параметры  #####################
WORK_DIR=/home/maniacello/nmt/en-ru-onmt # путь к репозиторию en-ru-onmt
ONMT_DIR=/home/maniacello/nmt/OpenNMT-py # путь к репозиторию OpenNMT-py
MODEL="checkpoint_step_48000.pt" # имя модели из каталога "models", используемой при переводе

GPU=0 # ID процессора, на котором должен быть выполнен перевод (-1 для перевода на CPU, список GPU можно вывести командой "nvidia-smi")
CPU_THREADS=6 # число ядер CPU, выделяемое для переводчика
BATCH=200 # Размер пакета (число предложений, обрабатываемых за один шаг). Значение указано для GPU с 16 ГБ памяти. Уменьшить при появлении ошибок о нехватке памяти.
BEAM=4
ALPHA=0.6
LENGTH_PENALTY=avg

SRC_LANG="en"
TGT_LANG="ru"
###############################################

set -e

if [ -f $1 ]; then 
  SRC=$1
else
  echo "Ошибка: неверное имя файла '"$1"'" 1>&2
  exit 1
fi

if [ -f $2 ]; then 
  echo "Ошибка: файл'"$2"' уже существует." 1>&2
  exit 1
else
  TGT=$2
fi

cd $WORK_DIR

TMP_DIR=$WORK_DIR/.tmp
if [ -d $TMP_DIR ]; then 
  rm -r $TMP_DIR
fi
mkdir $TMP_DIR

echo 'Токенизация...'
tools/tokenizer.perl -l $SRC_LANG -threads $CPU_THREADS < $SRC > $TMP_DIR/src.tok.txt

echo 'Сегментация...'
tools/apply_bpe.py -c tools/bpe_model.$SRC_LANG.txt < $TMP_DIR/src.tok.txt > $TMP_DIR/src.txt

echo 'Перевод...'
cd $ONMT_DIR
./translate.py -model $WORK_DIR/models/$MODEL \
  -src $TMP_DIR/src.txt \
  -output $TMP_DIR/tgt.txt \
  --gpu $GPU \
  --batch_size $BATCH \
  --beam_size $BEAM \
  --alpha $ALPHA \
  --length_penalty $LENGTH_PENALTY \
  --report_time \
  --verbose &> $WORK_DIR/translate.log

cd $WORK_DIR
echo 'Десегментация перевода...'
sed -r 's/(@@ )|(@@ ?$)//g' $TMP_DIR/tgt.txt > $TMP_DIR/tgt.tok.txt

echo 'Детокенизация перевода...'
tools/detokenizer.perl -l $TGT_LANG < $TMP_DIR/tgt.tok.txt > $TGT

rm -r $TMP_DIR

echo 'Перевод выполнен!'