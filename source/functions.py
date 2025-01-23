import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_edges(input_path, output_path):
    """
    Detect edges from a video by:
      1) Computing an average image of all frames.
      2) Applying threshold and morphological operations.
      3) Using Canny edge detection.
    """
    # Threshold / Morphology / Canny parameters
    THRESHOLD_VALUE = 10
    MORPH_KERNEL_SIZE = (3, 3)
    CANNY_MIN_VAL = 5
    CANNY_MAX_VAL = 150

    # Output file names
    OUTPUT_AVG_IMAGE = "1.avg_image.png"
    OUTPUT_BIN_MASK = "2.bin_mask.png"
    OUTPUT_EDGES = "3.edges.png"

    try:
        cap = cv2.VideoCapture(input_path)
        acc = None
        count = 0

        # Accumulate pixel intensities for the average image
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            if acc is None:
                acc = gray
            else:
                acc += gray

            count += 1

        cap.release()

        if count == 0:
            print("[Error] No frames read from video.")
            return None, None

        # Compute average image
        avg_img = (acc / count).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, OUTPUT_AVG_IMAGE), avg_img)

        # Threshold for binary mask
        _, bin_mask = cv2.threshold(avg_img, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(output_path, OUTPUT_BIN_MASK), bin_mask)

        # Morphological opening to clean up
        kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Canny edge detection
        edges = cv2.Canny(bin_mask, CANNY_MIN_VAL, CANNY_MAX_VAL)
        cv2.imwrite(os.path.join(output_path, OUTPUT_EDGES), edges)

        return edges, avg_img

    except Exception as e:
        print(f"[Error in detect_edges] {e}")
        return None, None


def hough_transform(edges, avg_img, output_path):
    """
    Apply the Probabilistic Hough Transform to detect lines.
    Draw the detected lines on the average image and save the result.
    """
    HOUGH_THRESHOLD = 5
    HOUGH_MIN_LINE_LENGTH = 30
    HOUGH_MAX_LINE_GAP = 1
    HOUGH_IMAGE_NAME = "4.hough_lines.png"

    try:
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LENGTH,
            maxLineGap=HOUGH_MAX_LINE_GAP
        )

        if lines is None:
            print("[Warning] No lines found in Hough transform.")
            return None, None

        # Convert average image to BGR for drawing
        line_img = cv2.cvtColor(avg_img, cv2.COLOR_GRAY2BGR)

        # Draw all detected lines
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (225, 225, 225), 1)

        cv2.imwrite(os.path.join(output_path, HOUGH_IMAGE_NAME), line_img)
        return lines, line_img

    except Exception as e:
        print(f"[Error in hough_transform] {e}")
        return None, None


def compute_line_params(x1, y1, x2, y2):
    """
    Compute slope (m), intercept (b), and type ('vertical' or 'normal') of a line.
    """
    try:
        if x2 == x1:
            # This is a vertical line
            return None, x1, "vertical"
        else:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return m, b, "normal"
    except Exception as e:
        print(f"[Error in compute_line_params] {e}")
        return None, None, None


def merge_lines(lines_params):
    """
    Merge lines that are similar in slope/intercept (for normal) or x-position (for vertical).
    """
    SLOPE_TOL = 0.1
    INTERCEPT_TOL = 20

    try:
        merged = []
        unmerged = lines_params[:]

        while unmerged:
            ref = unmerged.pop(0)
            m_ref, b_ref, kind_ref = ref
            same_group = [ref]
            remove_indices = []

            for i in range(len(unmerged)):
                m_cmp, b_cmp, kind_cmp = unmerged[i]
                if kind_cmp != kind_ref:
                    continue

                if kind_ref == "vertical":
                    # Merge if x-values are within intercept tolerance
                    if abs(b_cmp - b_ref) <= INTERCEPT_TOL:
                        same_group.append((m_cmp, b_cmp, kind_cmp))
                        remove_indices.append(i)
                else:
                    # Merge if slopes & intercepts are close
                    if (abs(m_cmp - m_ref) <= SLOPE_TOL and
                        abs(b_cmp - b_ref) <= INTERCEPT_TOL):
                        same_group.append((m_cmp, b_cmp, kind_cmp))
                        remove_indices.append(i)

            for idx in reversed(remove_indices):
                unmerged.pop(idx)

            # Average the parameters within the group
            if kind_ref == "vertical":
                x_vals = [ln[1] for ln in same_group]
                x_mean = sum(x_vals) / len(x_vals)
                merged.append((None, x_mean, "vertical"))
            else:
                m_vals = [ln[0] for ln in same_group]
                b_vals = [ln[1] for ln in same_group]
                m_mean = sum(m_vals) / len(m_vals)
                b_mean = sum(b_vals) / len(b_vals)
                merged.append((m_mean, b_mean, "normal"))

        return merged

    except Exception as e:
        print(f"[Error in merge_lines] {e}")
        return []


def lines_to_merged(lines):
    """
    Convert raw Hough lines to (slope, intercept, kind) and merge them.
    """
    try:
        if not lines:
            print("[Warning] No lines provided to lines_to_merged.")
            return []

        line_params = []
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            m, b, kind = compute_line_params(x1, y1, x2, y2)
            # Only keep valid lines
            if kind is not None:
                line_params.append((m, b, kind))

        # Merge lines
        return merge_lines(line_params)

    except Exception as e:
        print(f"[Error in lines_to_merged] {e}")
        return []


def draw_merged_lines(avg_img, merged_lines, output_path):
    """
    Draw merged lines on the average image and save the result.
    """
    MERGED_IMG_NAME = "5.merged_lines.png"

    try:
        line_img = cv2.cvtColor(avg_img, cv2.COLOR_GRAY2BGR)
        h, w = line_img.shape[:2]

        for (m, b, kind) in merged_lines:
            if kind == "vertical":
                x = int(round(b))
                if 0 <= x < w:
                    cv2.line(line_img, (x, 0), (x, h - 1), (255, 225, 225), 1)
            else:
                x1, x2 = 0, w - 1
                y1 = int(round(m * x1 + b))
                y2 = int(round(m * x2 + b))
                y1_clamped = max(0, min(h - 1, y1))
                y2_clamped = max(0, min(h - 1, y2))
                cv2.line(
                    line_img,
                    (x1, y1_clamped),
                    (x2, y2_clamped),
                    (255, 225, 225),
                    1
                )

        cv2.imwrite(os.path.join(output_path, MERGED_IMG_NAME), line_img)

    except Exception as e:
        print(f"[Error in draw_merged_lines] {e}")


def find_corners(merged_lines, img_shape):
    """
    Identify the two vertical and two horizontal lines, and return their
    intersection coordinates as (x1, x2, y1, y2).
    """
    try:
        h, w = img_shape[:2]
        verticals, horizontals = [], []

        # Separate lines by kind
        for (m, b, kind) in merged_lines:
            if kind == "vertical":
                verticals.append(b)
            else:
                horizontals.append(b)

        # We expect exactly 2 vertical and 2 horizontal lines
        if len(verticals) != 2 or len(horizontals) != 2:
            print("[Warning] We do not have exactly 2 vertical + 2 horizontal lines.")
            return None

        verticals.sort()
        horizontals.sort()

        x1 = int(round(verticals[0]))
        x2 = int(round(verticals[1]))
        y1 = int(round(horizontals[0]))
        y2 = int(round(horizontals[1]))

        return (x1, x2, y1, y2)

    except Exception as e:
        print(f"[Error in find_corners] {e}")
        return None


def put_label(img, text, pt):
    """
    Put text near a given point on the image (for corner labeling).
    """
    try:
        offset_x, offset_y = -10, -5
        text_pos = (pt[0] + offset_x, pt[1] + offset_y)
        cv2.putText(
            img,
            text,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 225, 225),
            1,
            cv2.LINE_AA
        )
    except Exception as e:
        print(f"[Error in put_label] {e}")


def visual_corner(merged_lines, avg_img, output_path):
    """
    Visualize the identified corners on the average image and save the result.
    Returns the corner points (b, c, d, e) if successfully identified.
    """
    CORNER_IMG_NAME = "6.labeled_corners.png"

    try:
        corners = find_corners(merged_lines, avg_img.shape)
        if corners is None:
            return None

        x1, x2, y1, y2 = corners
        print("Corners:", corners)

        labeled_img = cv2.cvtColor(avg_img, cv2.COLOR_GRAY2BGR)

        # Mark and label the four corners
        cv2.circle(labeled_img, (x1, y1), 1, (225, 225, 255), -1)
        cv2.circle(labeled_img, (x2, y1), 1, (225, 225, 255), -1)
        cv2.circle(labeled_img, (x1, y2), 1, (225, 225, 255), -1)
        cv2.circle(labeled_img, (x2, y2), 1, (225, 225, 255), -1)

        put_label(labeled_img, "a", (x1, y1))
        put_label(labeled_img, "b", (x2, y1))
        put_label(labeled_img, "f", (x1, y2))
        put_label(labeled_img, "e", (x2, y2))

        # Mark two additional points on the right edge
        right_end_top = (labeled_img.shape[1] - 1, y1)
        right_end_bottom = (labeled_img.shape[1] - 1, y2)
        cv2.circle(labeled_img, right_end_top, 1, (225, 255, 255), -1)
        cv2.circle(labeled_img, right_end_bottom, 1, (225, 255, 255), -1)

        put_label(labeled_img, "c", right_end_top)
        put_label(labeled_img, "d", right_end_bottom)

        cv2.imwrite(os.path.join(output_path, CORNER_IMG_NAME), labeled_img)

        # Return (b, c, d, e) in that order
        return (x2, y1), right_end_top, right_end_bottom, (x2, y2)

    except Exception as e:
        print(f"[Error in visual_corner] {e}")
        return None


def sample_line_intensity(frame, p1, p2, num_samples=50):
    """
    Sample pixel intensities along the line segment from p1 to p2
    and return the standard deviation of these intensities.
    """
    try:
        x1, y1 = p1
        x2, y2 = p2
        h, w = frame.shape[:2]

        xs = np.linspace(x1, x2, num_samples)
        ys = np.linspace(y1, y2, num_samples)
        values = []

        for i in range(num_samples):
            x, y = xs[i], ys[i]
            if 0 <= x < w and 0 <= y < h:
                xi, yi = int(round(x)), int(round(y))
                values.append(frame[yi, xi])

        return float(np.std(values)) if values else 0.0

    except Exception as e:
        print(f"[Error in sample_line_intensity] {e}")
        return 0.0


def measure_mixing_multi_lines(
    input_video,
    b, c, e, d,
    output_path,
    n_points=5,
    num_samples=50
):
    """
    Measure mixing by:
      1) Subdividing b->c and e->d into n_points segments.
      2) Connecting each subdivision (bc_points[i] ~ ed_points[i]) and computing
         the standard deviation of pixel intensities along each line.
      3) Annotating results on each frame of the video.
      4) Saving a time-series plot of all lines after processing every frame.
    """
    OUTPUT_VIDEO_NAME = "8.output.mp4"
    PLOT_IMG_NAME = "7.mixing_plot.png"

    try:
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"[Error] Cannot open input video: {input_video}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter(
            os.path.join(output_path, OUTPUT_VIDEO_NAME),
            fourcc, fps, (width, height)
        )

        bx, by = b
        cx, cy = c
        ex, ey = e
        dx, dy = d

        # 1) Subdivide b->c and e->d into n_points
        bc_points = []
        for i in range(n_points):
            t = i / (n_points - 1) if n_points > 1 else 0
            px = bx + (cx - bx) * t
            py = by + (cy - by) * t
            bc_points.append((px, py))

        ed_points = []
        for i in range(n_points):
            t = i / (n_points - 1) if n_points > 1 else 0
            qx = ex + (dx - ex) * t
            qy = ey + (dy - ey) * t
            ed_points.append((qx, qy))

        # List of standard deviation values for each line over time
        mixing_values = [[] for _ in range(n_points)]
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 2) Compute std dev for each subdivided line
            for i in range(n_points):
                p = bc_points[i]
                q = ed_points[i]
                line_std = sample_line_intensity(gray, p, q, num_samples)
                mixing_values[i].append(line_std)

                # Draw the line on the frame
                px, py = map(int, p)
                qx, qy = map(int, q)
                cv2.line(frame, (px, py), (qx, qy), (225, 255, 225), 1)

                # Label each line with its standard deviation
                mx = int(round((px + qx) / 2))
                my = int(round((py + qy) / 2))
                offset_y = -15 if i % 2 == 0 else 15
                text = f"L{i}:{line_std:.2f}"
                cv2.putText(
                    frame, text,
                    (mx - 50, my + offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (225, 225, 255),
                    1
                )

            out_writer.write(frame)
            frame_idx += 1

        cap.release()
        out_writer.release()

        # 3) Plot time series for each subdivided line
        plt.figure(figsize=(10, 5))
        x_axis = range(frame_idx)
        for i in range(n_points):
            plt.plot(x_axis, mixing_values[i], label=f"Line {i}")

        plt.title("Mixing (Std) for Each Line Over Frames")
        plt.xlabel("Frame")
        plt.ylabel("Std of Intensities")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_path, PLOT_IMG_NAME), dpi=150)
        plt.close()

    except Exception as e:
        print(f"[Error in measure_mixing_multi_lines] {e}")