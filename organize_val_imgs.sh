cd tiny-imagenet-200/val;
export IFS="	"
while read img class x1 y1 x2 y2; do
	if [ ! -d "$class" ]; then
		mkdir $class
	fi
done < val_annotations.txt

while read img class x1 y1 x2 y2; do
	mv "images/$img" $class
done < val_annotations.txt

rm -r images
