import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = 128
frame_size = N * N * 2  # number of double values per frame

# Get file size in bytes and convert to number of doubles
file_path = "output.bin"
file_size_bytes = os.path.getsize(file_path)
total_doubles = file_size_bytes // 8

print(f"File size in doubles: {total_doubles}")

# Determine number of frames
num_frames = total_doubles // frame_size
print(f"Found {num_frames} frames.")

# Read the data
data = np.fromfile(file_path, dtype=np.float64)

# If no frames (or only one), you won't be able to create a video animation.
if num_frames < 2:
    print("Warning: The file does not contain multiple frames. Only one frame was found.")

# Reshape data: shape = (num_frames, N, N, 2)
data = data.reshape((num_frames, N, N, 2))

# Create frames for magnitude and argument animations
abs_frames = []
arg_frames = []
for i in range(num_frames):
    # Reconstruct complex frame
    A_frame = data[i, :, :, 0] + 1j * data[i, :, :, 1]
    abs_frames.append(np.abs(A_frame))
    arg_frames.append(np.angle(A_frame))

# Create animation for magnitude if we have more than one frame
if num_frames > 1:
    fig_abs, ax_abs = plt.subplots()
    im_abs = ax_abs.imshow(abs_frames[0], cmap='viridis', origin='lower', vmin=0, vmax=np.max(np.stack(abs_frames)))
    cbar_abs = fig_abs.colorbar(im_abs, ax=ax_abs)
    ax_abs.set_title("Absolute Value of A")
    ax_abs.set_xlabel("x")
    ax_abs.set_ylabel("y")

    def update_abs(frame):
        im_abs.set_data(frame)
        return [im_abs]

    ani_abs = animation.FuncAnimation(fig_abs, update_abs, frames=abs_frames, interval=50, blit=True)
    ani_abs.save("abs_animation.mp4", writer='ffmpeg', fps=30)
    print("Magnitude animation saved as abs_animation.mp4")
else:
    print("Not enough frames for magnitude animation.")

# Create animation for argument (phase) if we have more than one frame
if num_frames > 1:
    fig_arg, ax_arg = plt.subplots()
    im_arg = ax_arg.imshow(arg_frames[0], cmap='twilight', origin='lower')
    cbar_arg = fig_arg.colorbar(im_arg, ax=ax_arg)
    ax_arg.set_title("Argument of A")
    ax_arg.set_xlabel("x")
    ax_arg.set_ylabel("y")

    def update_arg(frame):
        im_arg.set_data(frame)
        return [im_arg]

    ani_arg = animation.FuncAnimation(fig_arg, update_arg, frames=arg_frames, interval=50, blit=True)
    ani_arg.save("arg_animation.mp4", writer='ffmpeg', fps=30)
    print("Argument animation saved as arg_animation.mp4")
else:
    print("Not enough frames for argument animation.")

plt.show()