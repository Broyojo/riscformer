#!/bin/bash
# Build doomgeneric for the transformer CPU (rv32i bare metal).
set -e
cd "$(dirname "$0")"
CC=riscv64-elf-gcc
SRC=doomgeneric/doomgeneric
CFLAGS="-march=rv32i -mabi=ilp32 -O2 -fno-strict-aliasing -ffreestanding
  -DCMAP256 -DDOOMGENERIC_RESX=320 -DDOOMGENERIC_RESY=200
  -Iport/include -I$SRC -Iport -w"

OBJS=(dummy am_map doomdef doomstat dstrings d_event d_items d_iwad d_loop
  d_main d_mode d_net f_finale f_wipe g_game hu_lib hu_stuff info i_cdmus
  i_endoom i_joystick i_scale i_sound i_system i_timer memio m_argv m_bbox
  m_cheat m_config m_controls m_fixed m_menu m_misc m_random p_ceilng p_doors
  p_enemy p_floor p_inter p_lights p_map p_maputl p_mobj p_plats p_pspr
  p_saveg p_setup p_sight p_spec p_switch p_telept p_tick p_user r_bsp r_data
  r_draw r_main r_plane r_segs r_sky r_things sha1 sounds statdump st_lib
  st_stuff s_sound tables v_video wi_stuff w_checksum w_file w_main w_wad
  z_zone w_file_stdc i_input i_video doomgeneric)

mkdir -p build
for o in "${OBJS[@]}"; do
    if [ ! -f "build/$o.o" ] || [ "$SRC/$o.c" -nt "build/$o.o" ]; then
        echo "  CC $o.c"
        $CC $CFLAGS -c "$SRC/$o.c" -o "build/$o.o"
    fi
done
$CC $CFLAGS -c port/doomgeneric_tcpu.c -o build/dg_tcpu.o
$CC $CFLAGS -c port/syscalls.c -o build/syscalls.o
$CC $CFLAGS -c port/start.S -o build/start.o

echo "  LD doom.elf"
$CC $CFLAGS -c port/libmini.c -o build/libmini.o
$CC $CFLAGS -nostdlib -T port/link.ld \
    build/start.o build/dg_tcpu.o build/syscalls.o build/libmini.o \
    $(printf 'build/%s.o ' "${OBJS[@]}") \
    -lgcc -o build/doom.elf
riscv64-elf-objcopy -O binary build/doom.elf build/doom.bin
riscv64-elf-size build/doom.elf
ls -la build/doom.bin
