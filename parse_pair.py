# --------------------------------------------------------
# SMNet FaceNet
# Licensed under The MIT License [see LICENSE for details]
# Copyright 2019 smarsu. All Rights Reserved.
# --------------------------------------------------------

import os.path as osp


def main():
    pair_path = 'pair.txt'
    print('pair path:', pair_path)

    with open(pair_path, 'r') as fb:
        lines = fb.readlines()
        lines = lines[1:]

        lines = [line.strip().split() for line in lines]      

    persons = set()

    parsed_pair_path = 'parsed_' + pair_path
    print('parsed pair path:', parsed_pair_path)
    with open(parsed_pair_path, 'w') as fb:
        for line in lines:
            if len(line) == 3:
                fst_person_name = snd_person_name = line[0]
                fst_id = line[1].zfill(4)
                snd_id = line[2].zfill(4)
                match = 1
            elif len(line) == 4:
                fst_person_name = line[0]
                fst_id = line[1].zfill(4)
                snd_person_name = line[2]
                snd_id = line[3].zfill(4)
                match = 0

            fst_person_pth = osp.join(fst_person_name, 
                                      fst_person_name + '_' + fst_id + '.jpg')
            snd_person_pth = osp.join(snd_person_name, 
                                      snd_person_name + '_' + snd_id + '.jpg')    

            fb.write(fst_person_pth + ' ')
            fb.write(snd_person_pth + ' ')
            fb.write(str(match) + '\n')
        
            persons.add(fst_person_pth)
            persons.add(snd_person_pth)

    lfw_person_pth = 'lfw_person.txt'
    print('needed person path:', lfw_person_pth)
    with open(lfw_person_pth, 'w') as fb:
        for person in persons:
            fb.write(str(person) + '\n')


if __name__ == '__main__':
    main()
