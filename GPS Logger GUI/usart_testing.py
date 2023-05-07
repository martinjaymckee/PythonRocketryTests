import os
import os.path


def main():
    filename = 'teraterm.log'
    directory = 'testdata'
    correct_length = 64
    
    with open(os.path.join(directory, filename), 'r') as data:
        seq = 0
        last_error_seq = None
        gaps = []
        N = 0
        for line in data.readlines():
            line = line.strip()
            N += 1
            if not len(line) == correct_length:
                print('{} -> {}'.format(seq, len(line)))
                if last_error_seq is not None:
                    gap = seq - last_error_seq
                    gaps.append(gap)
                last_error_seq = seq
            seq += 1
        print('Gaps = {}'.format(gaps))
        print('Lines = {}'.format(N))
        
        
if __name__ == '__main__':
    main()