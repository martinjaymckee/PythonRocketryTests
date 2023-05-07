import matplotlib.pyplot as plt

import log_parser

if __name__ == '__main__':
    class NullEventHandler:
        def __init__(self):
            pass
        
        def ack(self):
            pass
            
        def nack(self):
            pass
        
        def finished(self):
            pass
            
        
    filename = 'testdata/test_4.csv'
    # gen = VirtualLogGenerator()
    # comms = VirtualCommChannel()
    event_handler = NullEventHandler()
    parser = log_parser.LogParser(event_handler, debug=True, allow_seq_gaps=True)
 
    samples = []
    with open(filename, 'r') as file:
        for line in file:
            new_samples = parser(line)
            samples += new_samples
            
    print('samples[0].sequence = {}'.format(samples[0].sequence_id))
    # for sample in samples:
    #     print(sample.sequence_id)
    print(parser.statistics)
        