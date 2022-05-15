from PIL import Image
import numpy as np

# usage of these libraries only for better interface purpose
from tqdm import tqdm
from itertools import product
import click
import os


def create_output_folder():
    try:
        os.mkdir('out')
    except OSError as error:
        pass


def as_mono(img):
    if len(img.shape) == 2:
        return img
    img = img.astype(np.float32)
    return (img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114).round().astype(np.uint8)


def as_logic(img):
    return img > img.mean()


@click.command()
@click.argument('file', type=click.Path())
@click.option('-m', '--mono', help='Treat image as black and white value?', is_flag=True)
@click.option('-p', '--points', help='Points to use for normalization, comma seperated', multiple=True,
              default=None)
def image_normalize(file, mono, points):
    create_output_folder()

    image_raw = Image.open(file)
    image = np.array(image_raw)

    if mono:
        image = as_mono(image).reshape((*image.shape[:2], 1))
    if points:
        points = [(0, 0), *[(int(a), int(b)) for a, b in [p.split(',') for p in points]], (255, 255)]
    else:
        points = [(0, 0)]
        while True:
            string_input = input('Provide next pair of points, seperated by comma (click Enter to finish): ')
            if string_input == '':
                break
            x, y = string_input.lstrip().rstrip().split(',')
            points.append((int(x), int(y)))
        points.append((255, 255))
    points = np.array(points)
    indices = [(points[:, 0] <= i).sum() - 1 for i in range(0, 256)]
    indices[-1] -= 1

    angles = [(points[i + 1, 1] - points[i, 1]) / (points[i + 1, 0] - points[i, 0]) for i in indices]
    LUT = np.array([round(sum(angles[:i])) for i in range(0, 256)])

    image = image.astype(np.uint32)
    if mono:
        intensity = image.reshape(image.shape[:2])
    else:
        intensity = ((image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3).round().astype(np.uint8)

    new_intensity = LUT[intensity]

    if mono:
        new_image = new_intensity
    else:
        new_image = np.zeros(image.shape)
        new_image[intensity != 0, 0] = image[intensity != 0, 0] / intensity[intensity != 0] * new_intensity[
            intensity != 0]
        new_image[intensity != 0, 1] = image[intensity != 0, 1] / intensity[intensity != 0] * new_intensity[
            intensity != 0]
        new_image[intensity != 0, 2] = image[intensity != 0, 2] / intensity[intensity != 0] * new_intensity[
            intensity != 0]

    new_image = (new_image - new_image.min()) / (new_image.max() - new_image.min()) * 255

    new_image = new_image.round().astype(np.uint8)
    new_image_raw = Image.fromarray(new_image)
    new_image_raw.save('out/normalize_o.jpg')
    new_image_raw.show('Result')


@click.command()
@click.argument('file', type=click.Path())
@click.argument('mask', type=click.IntRange(1, 10000000000000))
@click.option('-m', '--mono', help='Treat image as black and white value?', is_flag=True)
def image_filter(file, mask, mono):
    create_output_folder()

    image_raw = Image.open(file)
    image = np.array(image_raw)
    if mono:
        image = as_mono(image).reshape((*image.shape[:2], 1))

    width, height = image.shape[:2]

    mask_start = mask // 2
    mask_end = mask - mask_start

    new_image = np.zeros(image.shape[:2])

    positions = list(product(range(mask_start, width - mask_end), range(mask_start, height - mask_end)))

    for x, y in tqdm(positions):
        pixels = image[x - mask_start: x + mask_end, y - mask_start:y + mask_end, :]
        new_image[x, y] = pixels.std()

    new_image_raw = Image.fromarray(new_image.astype(np.uint8))
    new_image_raw.save('out/filter_o.jpg')
    new_image_raw.show('Result')


@click.command()
@click.argument('file', type=click.Path())
@click.argument('radius', type=click.IntRange(1, 10000000000000))
@click.option('-l', '--logic', help='Treat image as logic value?', is_flag=True)
def image_close(file, radius, logic):
    create_output_folder()

    image_raw = Image.open(file)
    image = np.array(image_raw).astype(np.uint8)
    image = as_mono(image)

    if logic:
        image = as_logic(image).astype(np.uint8) * 255

    Image.fromarray(image).show()

    width, height = image.shape[:2]
    lr, = np.ones(radius * 2 + 1).nonzero()
    X = lr.reshape(1, -1)
    Y = lr.reshape(-1, 1)

    mask = (np.sqrt((X - radius) ** 2 + (Y - radius) ** 2) <= radius)

    mask_start = radius
    mask_end = radius

    new_image = np.zeros(image.shape).astype(np.uint8)
    if logic:
        new_image = new_image.astype(np.bool)

    positions = list(product(range(mask_start, width - mask_end), range(mask_start, height - mask_end)))

    for x, y in tqdm(positions):
        pixels = image[x - mask_start: x + mask_end + 1, y - mask_start:y + mask_end + 1]

        new_image[x, y] = pixels[mask].min()
    t2 = new_image.copy()
    for x, y in tqdm(positions):
        pixels = t2[x - mask_start: x + mask_end + 1, y - mask_start:y + mask_end + 1]
        new_image[x, y] = pixels[mask].max()

    new_image_raw = Image.fromarray(new_image)
    new_image_raw.save('out/close_o.jpg')
    new_image_raw.show('Result')


@click.command()
@click.argument('file', type=click.Path())
def image_convex(file):
    create_output_folder()

    image_raw = Image.open(file)
    image = np.array(image_raw).astype(np.uint8)
    image = as_mono(image)
    image = as_logic(image)

    Image.fromarray(image).show()

    width, height = image.shape[:2]

    td, lr = np.ones((width, height)).nonzero()

    td = td.reshape(width, height)
    lr = lr.reshape(width, height)

    positions = np.dstack((td, lr))

    valid_positions = positions[image].reshape(-1, 2)

    min_x_i = valid_positions[:, 0].argmin()
    max_x_i = valid_positions[:, 0].argmax()

    min_x = valid_positions[min_x_i]
    max_x = valid_positions[max_x_i]

    angles = np.cross(max_x - min_x, valid_positions - min_x)

    def rec(points, A, B, i=0):
        if len(points) == 0:
            return []
        la = B[1] - A[1]
        lb = B[0] - A[0]
        lc = A[0] * la - lb * A[1]

        distances = abs(points[:, 0] * la - points[:, 1] * lb - lc) / np.sqrt(la ** 2 + lb ** 2)

        max_d_i = distances.argmax()
        max_d = points[max_d_i]

        angles_A = np.cross(max_d - A, points - A)
        angles_B = np.cross(B - max_d, points - max_d)
        return [*rec(points[angles_B < 0], max_d, B, i + 1), max_d, *rec(points[angles_A < 0], A, max_d, i + 1)]

    hull = [min_x, *rec(valid_positions[angles > 0], max_x, min_x), max_x,
            *rec(valid_positions[angles < 0], min_x, max_x)]

    positions_list = positions.reshape((-1, 2))

    counts = np.ones(len(positions_list)).astype(np.uint32)

    for A, B in zip(np.roll(hull, 1, axis=0), hull):
        counts *= np.cross(B - A, positions_list - A) <= 0

    Image.fromarray(image).show()
    new_image_raw = Image.fromarray(counts.reshape(image.shape) > 0)
    new_image_raw.save('out/convex_o.jpg')
    new_image_raw.show('Result')
