from pathlib import Path
from tqdm import tqdm
import urllib.request
from epub_conversion.utils import open_book, convert_epub_to_lines
Michael Makukha, [15.08.19 20:01]
from bs4 import BeautifulSoup


BOOKS = [
    'http://static.flibusta.site/converter/get/convert?url=http%3A%2F%2Fflibusta.is%2Fb%2F2690781d8d4e4e6f98fda550c3c1146a%2Fdownload&out=epub&md5=2690781d8d4e4e6f98fda550c3c1146a',
    'http://static.flibusta.site/converter/get/convert?url=http%3A%2F%2Fflibusta.is%2Fb%2F32c951a3abca50ed3f794a9d2336e6e5%2Fdownload&out=epub&md5=32c951a3abca50ed3f794a9d2336e6e5',
    'http://static.flibusta.site/converter/get/convert?url=http%3A%2F%2Fflibusta.is%2Fb%2F5cd2c872ec978ba71440ebf06d83a84a%2Fdownload&out=epub&md5=5cd2c872ec978ba71440ebf06d83a84a',
    'http://static.flibusta.site/converter/get/convert?url=http%3A%2F%2Fflibusta.is%2Fb%2Fafc8cc4fd385057372e8e1c019029e97%2Fdownload&out=epub&md5=afc8cc4fd385057372e8e1c019029e97',
    'http://static.flibusta.site/converter/get/convert?url=http%3A%2F%2Fflibusta.is%2Fb%2F65ae5b25c60dad951779d38f3aade813%2Fdownload&out=epub&md5=65ae5b25c60dad951779d38f3aade813',
    'http://static.flibusta.site/converter/get/convert?url=http%3A%2F%2Fflibusta.is%2Fb%2F3c73b9172433951cc70c07b7ea439769%2Fdownload&out=epub&md5=3c73b9172433951cc70c07b7ea439769',
    'http://static.flibusta.site/converter/get/convert?url=http%3A%2F%2Fflibusta.is%2Fb%2F89ffd2b1ebbe3d641ef695637ad5e869%2Fdownload&out=epub&md5=89ffd2b1ebbe3d641ef695637ad5e869',
    'http://static.flibusta.site/converter/get/convert?url=http%3A%2F%2Fflibusta.is%2Fb%2F54bfae4c0e70691469f6d82a5ba4e311%2Fdownload&out=epub&md5=54bfae4c0e70691469f6d82a5ba4e311',
    'http://static.flibusta.site/converter/get/convert?url=http%3A%2F%2Fflibusta.is%2Fb%2F831db917512b3ffb2f0246271caff7af%2Fdownload&out=epub&md5=831db917512b3ffb2f0246271caff7af',
    ]
BUILDDIR = Path('build')


def download():
    """Download files."""

    BUILDDIR.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(tqdm(BOOKS)):
        with urllib.request.urlopen(url) as f:
            (BUILDDIR/f'{i}.epub').write_bytes(f.read())

def train():
    """Train the model."""

    for path in BUILDDIR.glob('*.epub'):
        book = open_book(path)
        lines = convert_epub_to_lines(book)
        for line in lines:
            soup = BeautifulSoup(line)
            text = soup.get_text()
            breakpoint()

if __name__ == '__main__':
    model = train()