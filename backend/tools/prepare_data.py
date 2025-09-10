import cv2, glob, urllib.request
from pathlib import Path

IMG_URLS = [
 "https://tse1.mm.bing.net/th/id/OIP.vFkkFDeIN6FrbZpx78rZ2gHaE8?w=474&h=474&c=7&p=0",
 "https://tse1.mm.bing.net/th/id/OIP.PBA5fyzcbYtB0x44PryliAHaHa?w=474&h=474&c=7&p=0",
 "https://tse2.mm.bing.net/th/id/OIP.U0OY5NUaEtRy04TlZvOVWAHaHa?w=474&h=474&c=7&p=0",
 "https://tse3.mm.bing.net/th/id/OIP.YCowa3Npw3ddstrClABxIQHaFb?w=474&h=474&c=7&p=0",
]

OUT = Path("datasets/dials/images/train")
OUT.mkdir(parents=True, exist_ok=True)

def download_images():
    for i,u in enumerate(IMG_URLS):
        fp = OUT / f"seed_{i:02d}.jpg"
        try:
            urllib.request.urlretrieve(u, fp.as_posix())
            print("Saved", fp)
        except Exception as e:
            print("Fail", u, e)

def sample_video_frames(video_glob="sample_videos/*.mp4", every_n=30):
    for vid in glob.glob(video_glob):
        cap = cv2.VideoCapture(vid); n=0; idx=0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if n % every_n == 0:
                (OUT / f"{Path(vid).stem}_f{idx:06d}.jpg").write_bytes(
                    cv2.imencode(".jpg", frame)[1].tobytes()
                ); idx += 1
            n += 1
        cap.release()

if __name__ == "__main__":
    download_images()
    sample_video_frames()
