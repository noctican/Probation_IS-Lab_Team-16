"""GUI manual untuk memilih gambar duplikat yang di-KEEP / di-REMOVE.

Menampilkan satu grup duplikat per layar, foto-foto anggota grup disandingkan
side-by-side agar bisa dibandingkan. Per gambar bisa dipilih KEEP atau tidak
(tidak dipilih = REMOVE). Boleh tidak memilih apa pun (semua dibuang) atau
memilih lebih dari satu.

Sumber grup (rekonstruksi dedupe v2, MD5 identik):
  outputs/results/duplicate_groups_md5.csv
  (fallback: outputs/results/duplicate_groups_summary.csv)

Jalankan:
  python "Reproduce Code/dedupe_gui.py"
atau (disarankan, jika butuh Pillow/ImageTk dari venv):
  .venv/bin/python "Reproduce Code/dedupe_gui.py"

Navigasi: Prev/Next (atau panah kiri/kanan), klik tombol pada tiap gambar
untuk toggle KEEP/REMOVE. Ctrl+S untuk simpan.

Keputusan disimpan ke:
  outputs/results/keep_decisions.json
  outputs/results/duplicate_groups_with_decision.csv
Jika keep_decisions.json sudah ada, keputusan sebelumnya dimuat (resume).
"""

from __future__ import annotations

import base64
import csv
import io
import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:  # pragma: no cover
    Image = None
    ImageTk = None

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "Datasets" / "images" / "images"
RESULTS = ROOT / "outputs" / "results"
MEMBERS_CSV = RESULTS / "duplicate_groups_md5.csv"
GROUPS_CSV = RESULTS / "duplicate_groups_summary.csv"
DECISIONS_JSON = RESULTS / "keep_decisions.json"
DECISIONS_CSV = RESULTS / "duplicate_groups_with_decision.csv"

THUMB_SIZE = (360, 360)


def natural_path_key(rel_path: str):
    """Urutkan natural berdasarkan nama file, mis. Image_3 < Image_12 < Image_23."""
    name = Path(rel_path).name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def load_groups() -> list[dict]:
    """Load duplicate groups.

    Each group: {gid, md5, members: [{rel_path,class_name,source_group}], kept: set(rel_path), cross_class: bool}

    Aturan:
    - Grup 1-kelas (semua anggota di class_name yang sama): otomatis KEEP 1
      (nama file natural terkecil), sisanya REMOVE, tidak ditampilkan di GUI.
    - Grup lintas-kelas (ada anggota di kelas berbeda): ditampilkan untuk
      diputuskan manual. Default KEEP = member role 'kept' (jika ada) atau
      nama file natural terkecil. Jika keep_decisions.json ada, keputusan
      tersimpan untuk grup lintas-kelas dimuat (resume).
    """
    groups: dict[str, dict] = {}
    order: list[str] = []

    if MEMBERS_CSV.exists():
        with MEMBERS_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gid = str(row["group_id"])
                if gid not in groups:
                    groups[gid] = {"gid": gid, "md5": row.get("md5", ""), "members": [], "kept": set()}
                    order.append(gid)
                groups[gid]["members"].append(
                    {
                        "rel_path": row["rel_path"],
                        "class_name": row.get("class_name", ""),
                        "source_group": row.get("source_group", ""),
                        "role": row.get("member_role", ""),
                    }
                )
    elif GROUPS_CSV.exists():
        with GROUPS_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gid = str(row["group_id"])
                all_paths = [p.strip() for p in row.get("all_rel_paths", "").split("|") if p.strip()]
                members = []
                for p in all_paths:
                    parts = Path(p).parts
                    cls = parts[0] if parts else row.get("kept_class_name", "")
                    src = parts[1] if len(parts) > 1 else row.get("kept_source_group", "")
                    members.append({"rel_path": p, "class_name": cls, "source_group": src})
                groups[gid] = {"gid": gid, "md5": row.get("md5", ""), "members": members, "kept": set()}
                order.append(gid)
    else:
        raise FileNotFoundError(
            f"Tidak menemukan data grup duplikat.\n"
            f"Dibutuhkan: {MEMBERS_CSV} atau {GROUPS_CSV}"
        )

    result = [groups[gid] for gid in order]
    for g in result:
        g["members"].sort(key=lambda m: natural_path_key(m["rel_path"]))
        classes = {m["class_name"] for m in g["members"]}
        g["cross_class"] = len(classes) > 1
        if g["cross_class"]:
            kept = [m["rel_path"] for m in g["members"] if m.get("role") == "kept"]
            if not kept:
                kept = [g["members"][0]["rel_path"]]
            g["kept"] = set(kept)
        else:
            # sama di dalam 1 kelas: simpan 1 saja (nama file natural terkecil), sisanya dibuang
            g["kept"] = {g["members"][0]["rel_path"]} if g["members"] else set()

    # Resume keputusan tersimpan hanya untuk grup lintas-kelas
    if DECISIONS_JSON.exists():
        try:
            payload = json.loads(DECISIONS_JSON.read_text(encoding="utf-8"))
            saved = {str(item.get("group_id")): set(item.get("keep", [])) for item in payload.get("groups", [])}
            for g in result:
                if g.get("cross_class") and g["gid"] in saved:
                    member_paths = {m["rel_path"] for m in g["members"]}
                    g["kept"] = saved[g["gid"]] & member_paths
        except Exception:
            pass

    return result


def resolve_image_path(rel_path: str) -> Path:
    # rel_path seperti "aerosol_cans/default/Image_133.png" relatif ke DATA_ROOT
    p = DATA_ROOT / rel_path
    if p.exists():
        return p
    # fallback: cari berdasarkan nama file di dalam folder kelas (jaga-jaga path beda)
    name = Path(rel_path).name
    parts = Path(rel_path).parts
    if parts:
        candidate = DATA_ROOT / parts[0]
        if candidate.exists():
            for hit in candidate.rglob(name):
                return hit
    return p


def make_photoimage(path: Path, max_size=THUMB_SIZE):
    """Return a Tk-compatible image object for path, resized to fit max_size.

    Mengembalikan (photo, None) atau (None, errstr).
    - Pillow+ImageTk: thumbnail PhotoImage.
    - Pillow saja: ("pngdata", bytes) untuk tk.PhotoImage(data=...).
    - Tanpa Pillow: coba tk.PhotoImage(file=...) untuk PNG/GIF (tanpa resize).
    """
    if Image is not None:
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail(max_size, Image.Resampling.LANCZOS)
                if ImageTk is not None:
                    return ImageTk.PhotoImage(im), None
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                return ("pngdata", base64.b64encode(buf.getvalue())), None
        except Exception as e:  # pragma: no cover
            return None, str(e)

    # Fallback tanpa Pillow: Tk 8.6+/9 mendukung PNG/GIF via file/data (ukuran asli)
    if path.suffix.lower() in {".png", ".gif", ".ppm", ".pgm"}:
        try:
            return tk.PhotoImage(file=str(path)), None
        except Exception as e:  # pragma: no cover
            return None, str(e)
    return None, "Pillow (PIL) tidak tersedia untuk format ini"


class DedupeReviewerApp:
    def __init__(self, root: tk.Tk, all_groups: list[dict]):
        self.root = root
        self.all_groups = all_groups
        self.groups = [g for g in all_groups if g.get("cross_class")]  # hanya lintas-kelas
        self.index = 0
        self._photo_refs: list = []  # cegah garbage-collection

        root.title("Dedupe Reviewer — pilih KEEP / REMOVE per gambar (lintas-kelas)")
        root.geometry("1200x760")
        root.minsize(900, 600)

        self._build_ui()
        if self.groups:
            self._show_group(0)
        else:
            self._show_empty()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side="top", fill="x")

        self.info_label = ttk.Label(top, text="", font=("TkDefaultFont", 10, "bold"))
        self.info_label.pack(side="left")

        nav = ttk.Frame(top)
        nav.pack(side="right")
        ttk.Button(nav, text="◀ Prev", command=self.prev_group).pack(side="left", padx=2)
        ttk.Button(nav, text="Next ▶", command=self.next_group).pack(side="left", padx=2)

        meta = ttk.Frame(self.root, padding=(8, 0))
        meta.pack(side="top", fill="x")
        self.meta_label = ttk.Label(meta, text="", foreground="#444")
        self.meta_label.pack(side="left")

        hint = ttk.Frame(self.root, padding=(8, 0))
        hint.pack(side="top", fill="x")
        auto_n = sum(1 for g in self.all_groups if not g.get("cross_class"))
        ttk.Label(
            hint,
            text=(
                "Hanya grup lintas-kelas (isi sama, kelas berbeda) yang ditampilkan. "
                f"Grup 1-kelas ({auto_n} grup) otomatis disimpan 1 per grup dan disembunyikan. "
                "Klik tombol pada tiap gambar untuk toggle KEEP/REMOVE. "
                "Tidak ada yang dipilih = semua dibuang. Bisa pilih lebih dari satu."
            ),
            foreground="#555",
            wraplength=1100,
            justify="left",
        ).pack(side="left", anchor="w")

        # area gambar (scrollable horizontal bila perlu)
        container = ttk.Frame(self.root)
        container.pack(side="top", fill="both", expand=True, padx=8, pady=8)

        self.canvas = tk.Canvas(container, highlightthickness=0)
        hbar = ttk.Scrollbar(container, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=hbar.set)
        self.canvas.pack(side="top", fill="both", expand=True)
        hbar.pack(side="bottom", fill="x")

        self.images_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.images_frame, anchor="nw")
        self.images_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(side="bottom", fill="x")
        ttk.Button(bottom, text="Simpan keputusan", command=self.save_decisions).pack(side="left", padx=4)
        ttk.Button(bottom, text="Default grup ini (nama terkecil)", command=self.reset_current_default).pack(side="left", padx=4)
        ttk.Button(bottom, text="Kosongkan grup ini (buang semua)", command=self.clear_current).pack(side="left", padx=4)
        ttk.Button(bottom, text="Default semua grup", command=self.reset_all_defaults).pack(side="left", padx=4)

        self.status_label = ttk.Label(bottom, text="", foreground="#333")
        self.status_label.pack(side="right")

        self.root.bind("<Left>", lambda e: self.prev_group())
        self.root.bind("<Right>", lambda e: self.next_group())
        self.root.bind("<Control-s>", lambda e: self.save_decisions())

    def _on_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # buat frame minimal setinggi canvas
        self.canvas.itemconfig(self.canvas_window, height=event.height)

    def _show_empty(self) -> None:
        self.info_label.config(text="Tidak ada grup lintas-kelas untuk ditinjau")
        self.meta_label.config(text="")
        for w in self.images_frame.winfo_children():
            w.destroy()
        ttk.Label(
            self.images_frame,
            text=(
                "Semua grup duplikat berada dalam 1 kelas yang sama.\n"
                "Tiap grup otomatis disimpan 1 gambar (nama file natural terkecil), sisanya dibuang.\n"
                "Klik 'Simpan keputusan' untuk menulis hasil."
            ),
            justify="center",
            padding=40,
        ).grid(row=0, column=0, padx=20, pady=20)
        self.status_label.config(
            text=f"0 grup lintas-kelas | {len(self.all_groups)} grup 1-kelas (auto)"
        )

    def _current_group(self) -> dict:
        return self.groups[self.index]

    def _show_group(self, index: int) -> None:
        if not self.groups:
            self._show_empty()
            return
        self.index = max(0, min(index, len(self.groups) - 1))
        g = self._current_group()

        self.info_label.config(
            text=f"Grup lintas-kelas {self.index + 1} / {len(self.groups)}  (id {g['gid']}, {len(g['members'])} gambar)"
        )
        self.meta_label.config(text=f"MD5: {g['md5']}")
        self._update_status()

        for w in self.images_frame.winfo_children():
            w.destroy()
        self._photo_refs.clear()

        for col, member in enumerate(g["members"]):
            panel = ttk.LabelFrame(
                self.images_frame,
                text=Path(member["rel_path"]).name,
                padding=8,
            )
            panel.grid(row=0, column=col, padx=8, pady=8, sticky="n")

            img_path = resolve_image_path(member["rel_path"])
            photo_obj, err = make_photoimage(img_path, THUMB_SIZE)
            photo = None
            if photo_obj is not None:
                if isinstance(photo_obj, tuple) and photo_obj[0] == "pngdata":
                    try:
                        photo = tk.PhotoImage(data=photo_obj[1])
                    except Exception as e:
                        err = str(e)
                        photo = None
                else:
                    photo = photo_obj
            if photo is not None:
                self._photo_refs.append(photo)
                img_label = ttk.Label(panel, image=photo)
            else:
                msg = "(gambar tidak dapat dimuat)"
                if err:
                    msg += f"\n{err}"
                if not img_path.exists():
                    msg = f"(file tidak ditemukan)\n{img_path}"
                img_label = ttk.Label(panel, text=msg, width=40, anchor="center", justify="center")
            img_label.pack()

            ttk.Label(panel, text=member["rel_path"], wraplength=340, justify="center").pack(pady=(6, 0))
            ttk.Label(
                panel,
                text=f"class: {member['class_name']}   |   source: {member['source_group']}",
                foreground="#333",
            ).pack()

            btn = tk.Button(panel, width=18, relief="raised")
            btn.pack(pady=8)

            def make_toggle(m=member, grp=g, b=btn):
                def _toggle():
                    rp = m["rel_path"]
                    if rp in grp["kept"]:
                        grp["kept"].discard(rp)
                    else:
                        grp["kept"].add(rp)
                    self._style_toggle(b, rp in grp["kept"])
                    self._update_status()

                return _toggle

            btn.configure(command=make_toggle())
            self._style_toggle(btn, member["rel_path"] in g["kept"])

        self.images_frame.update_idletasks()
        self._on_frame_configure()
        self.canvas.xview_moveto(0)

    def _style_toggle(self, btn: tk.Button, is_keep: bool) -> None:
        if is_keep:
            btn.config(
                text="✓ KEEP",
                bg="#2e7d32",
                fg="white",
                activebackground="#1b5e20",
                activeforeground="white",
            )
        else:
            btn.config(
                text="✗ REMOVE",
                bg="#c62828",
                fg="white",
                activebackground="#8e0000",
                activeforeground="white",
            )

    def _update_status(self) -> None:
        if not self.groups:
            return
        g = self._current_group()
        kept_n = len(g["kept"])
        self.status_label.config(
            text=f"Dipilih KEEP: {kept_n}/{len(g['members'])} gambar di grup ini"
        )

    def prev_group(self) -> None:
        if self.groups and self.index > 0:
            self._show_group(self.index - 1)

    def next_group(self) -> None:
        if self.groups and self.index < len(self.groups) - 1:
            self._show_group(self.index + 1)

    def reset_current_default(self) -> None:
        if not self.groups:
            return
        g = self._current_group()
        first = min(g["members"], key=lambda m: natural_path_key(m["rel_path"]))["rel_path"]
        g["kept"] = {first}
        self._show_group(self.index)

    def clear_current(self) -> None:
        if not self.groups:
            return
        g = self._current_group()
        g["kept"] = set()
        self._show_group(self.index)

    def reset_all_defaults(self) -> None:
        if not self.groups:
            return
        if not messagebox.askyesno(
            "Konfirmasi",
            "Kembalikan semua grup lintas-kelas yang ditampilkan ke default (nama file terkecil = KEEP)?",
        ):
            return
        for g in self.groups:
            first = min(g["members"], key=lambda m: natural_path_key(m["rel_path"]))["rel_path"]
            g["kept"] = {first}
        self._show_group(self.index)

    def save_decisions(self) -> None:
        RESULTS.mkdir(parents=True, exist_ok=True)
        payload = {
            "groups": [
                {"group_id": g["gid"], "md5": g["md5"], "keep": sorted(g["kept"])}
                for g in self.all_groups
            ]
        }
        DECISIONS_JSON.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        with DECISIONS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["md5", "rel_path", "class_name", "source_group", "decision"]
            )
            writer.writeheader()
            for g in self.all_groups:
                for m in g["members"]:
                    decision = "keep" if m["rel_path"] in g["kept"] else "remove"
                    writer.writerow(
                        {
                            "md5": g["md5"],
                            "rel_path": m["rel_path"],
                            "class_name": m["class_name"],
                            "source_group": m["source_group"],
                            "decision": decision,
                        }
                    )

        kept_total = sum(len(g["kept"]) for g in self.all_groups)
        removed_total = sum(len(g["members"]) - len(g["kept"]) for g in self.all_groups)
        auto_n = sum(1 for g in self.all_groups if not g.get("cross_class"))
        review_n = sum(1 for g in self.all_groups if g.get("cross_class"))
        msg = (
            f"Keputusan disimpan:\n"
            f"- {DECISIONS_JSON}\n"
            f"- {DECISIONS_CSV}\n\n"
            f"KEEP: {kept_total} gambar, REMOVE: {removed_total} gambar "
            f"(dari {len(self.all_groups)} grup total).\n"
            f"Grup lintas-kelas (ditinjau): {review_n}, grup 1-kelas (auto 1 keep): {auto_n}."
        )
        self.status_label.config(text=f"Tersimpan. KEEP {kept_total} / REMOVE {removed_total}.")
        messagebox.showinfo("Tersimpan", msg)


def main() -> None:
    groups = load_groups()
    root = tk.Tk()
    # style dasar
    try:
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app = DedupeReviewerApp(root, groups)
    root.mainloop()


if __name__ == "__main__":
    main()
