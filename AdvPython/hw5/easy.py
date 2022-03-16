#!/usr/bin/env python

import asyncio
import os
from argparse import ArgumentParser
from uuid import uuid4

import aiohttp


async def downsave_img(url, session, savepath):
    async with session.get(url) as r:
        out = await r.read()
        with open(savepath, "wb+") as savefile:
            savefile.write(out)


async def download(url="https://picsum.photos/200", n=1, savedir="artifacts/easy"):
    async with aiohttp.ClientSession() as ses:
        tasks = []
        for _ in range(n):
            savepath = os.path.join(savedir, str(uuid4()) + ".png")  # somewhat unique name
            tasks.append(asyncio.create_task(downsave_img(url, ses, savepath)))

        # asyncio.run(await asyncio.gather(*tasks))  # this is not working...
        return await asyncio.gather(*tasks)


def main():
    parser = ArgumentParser()
    parser.add_argument("-n", type=int, default=1, help="Number of files to download")
    parser.add_argument(
        "-sd", "--savedir", type=str, default="artifacts/easy", help="Path to where downloaded files are stored"
    )
    parser.add_argument("--url", type=str, default="https://picsum.photos/200")

    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    # download(url=args.url, n=args.n, savedir=args.savedir)
    asyncio.run(download(url=args.url, n=args.n, savedir=args.savedir))


if __name__ == "__main__":
    main()
