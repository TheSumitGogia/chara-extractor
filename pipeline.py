import argparse

'''Run pipeline for character extraction classifier creation and testing
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='run character extractor creation/testing pipeline')
    parser.add_argument('-s', '--steps', default=['collect', 'labeling', 'features', 'learning', 'evaluate'], nargs='+', help='particular pipeline steps to run')
