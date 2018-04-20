"""
Download dataset using official Kaggle API.

Credentials to access Kaggle should be placed into ~/.kaggle/kaggle.json.

Note that rules of competition still should be accepted via Kaggle's UI before
dataset could be downloaded.
"""
import os
import sys
import time
import glob
import tarfile
import zipfile
import argparse
from subprocess import Popen, PIPE


def download_dataset(name, output, timeout=600):
    """Downloads dataset using Kaggle API."""
    if output is None:
        home = os.path.expanduser('~/data/')
        folder = os.path.expandvars(os.environ.get('KAGGLE_DATASETS', home))
        output = os.path.join(folder, name)

    script = os.path.join(os.path.dirname(sys.executable), 'kaggle')
    cmd = f'{script} competitions download -c {name} -p {output}'
    process = Popen(cmd, stdout=PIPE, bufsize=1, close_fds=True, shell=True)
    time_limit = time.time() + timeout
    print('Downloading dataset into folder', output)

    while True:
        try:
            if time.time() >= time_limit:
                sys.exit(1)

            if process.poll() is None:
                raw_output = process.stdout.readline()
                decoded = raw_output.decode(encoding='utf-8')
                if not decoded:
                    continue
                print(decoded.strip('\n'))
            else:
                print('Downloading process was finished')
                break

        except KeyboardInterrupt:
            process.terminate()
            print('Keyboard interrupt. Terminating...')
            break

        except Exception as e:
            process.terminate()
            print('Unexpected error: %s', e)
            break

    path = os.path.abspath(output)
    print('Dataset path:', path)
    return path


def unzip_archives(folder):
    """Unzips downloaded datasets if any of them is provided """

    def is_archive(path):
        return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)

    for filename in glob.glob(folder + '/*'):
        if is_archive(filename):
            print('Unpacking file:', filename)
        if zipfile.is_zipfile(filename):
            with zipfile.ZipFile(filename) as arch:
                arch.extractall(path=folder)
        elif tarfile.is_tarfile(filename):
            with tarfile.open(filename) as arch:
                arch.extractall(path=folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--name', required=True, help='Competition name')
    parser.add_argument(
        '-p', '--path', required=False, help='Path to save competition files')
    args = parser.parse_args()

    if args.path is not None and os.path.exists(args.path):
        print('Warning: path already exists:', args.path, 'Terminating...')
        sys.exit(1)

    output_path = download_dataset(args.name, args.path)
    unzip_archives(output_path)


if __name__ == '__main__':
    main()
