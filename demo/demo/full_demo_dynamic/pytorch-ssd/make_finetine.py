import glob
from collections import defaultdict
root = 'data/voc0712/test/VOC2007/ImageSets/Main'
all_samples = open(f'{root}/test.txt').read().splitlines()


img2class = defaultdict(list)

for f in glob.glob(f'{root}/*_test*'):
    name = f.split('_')[0].split('/')[-1]

    samples = open(f).read().splitlines()
    for sam in samples:
        img_id, flag = sam.split()
        if flag != '-1':
            img2class[img_id].append(name)
    
allowed_classes = ['chair', 'person', 'tvmonitor']
ft_samples = []
for img, classes in img2class.items():
    if set(classes) <= set(allowed_classes):
        ft_samples.append(img)
        print(img)
ft_samples = list(set(ft_samples))
out_str = '\n'.join(sorted(ft_samples))
with open(f'{root}/finetune.txt', 'w') as f:
    f.write(out_str)
