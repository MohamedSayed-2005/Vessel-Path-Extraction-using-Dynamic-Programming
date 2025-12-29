from PIL import Image, ImageDraw, ImageTk
import numpy as np
from scipy import ndimage
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading


class VesselPathExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vessel Path Extraction - Dynamic Programming")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.img_array = None
        self.gray_img = None
        self.result_img = None
        self.start_row = None
        self.reduced_array = None
        self.path_original = None
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🏥 Vessel Path Extraction using Dynamic Programming",
                              font=('Arial', 18, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=15)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_container, bg='white', relief='raised', borderwidth=2)
        left_panel.pack(side='left', fill='y', padx=(0, 5))
        
        # Control sections
        self.create_file_controls(left_panel)
        self.create_parameter_controls(left_panel)
        self.create_action_controls(left_panel)
        self.create_info_display(left_panel)
        
        # Right panel - Image display
        right_panel = tk.Frame(main_container, bg='white', relief='raised', borderwidth=2)
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.create_image_display(right_panel)
        
    def create_file_controls(self, parent):
        frame = tk.LabelFrame(parent, text="📁 File Operations", font=('Arial', 11, 'bold'),
                             bg='white', fg='#2c3e50', padx=10, pady=10)
        frame.pack(fill='x', padx=10, pady=10)
        
        self.load_btn = tk.Button(frame, text="Load Image", command=self.load_image,
                                 bg='#0078D4', fg='white', font=('Arial', 10, 'bold'),
                                 relief='raised', cursor='hand2', width=20, height=2, activebackground='#005A9E', activeforeground='white', disabledforeground='white')
        self.load_btn.pack(pady=5)
        
        self.save_btn = tk.Button(frame, text="Save Result", command=self.save_result,
                                 bg='#107C10', fg='white', font=('Arial', 10, 'bold'),
                                 relief='raised', cursor='hand2', width=20, height=2,
                                 state='disabled', activebackground='#0B5F0B', activeforeground='white', disabledforeground='white')
        self.save_btn.pack(pady=5)
        
    def create_parameter_controls(self, parent):
        frame = tk.LabelFrame(parent, text="⚙️ Parameters", font=('Arial', 11, 'bold'),
                             bg='white', fg='#2c3e50', padx=10, pady=10)
        frame.pack(fill='x', padx=10, pady=10)
        
        # Block Size
        tk.Label(frame, text="Block Size:", bg='white', font=('Arial', 9)).pack(anchor='w', pady=(5,0))
        self.block_size_var = tk.IntVar(value=2)
        block_frame = tk.Frame(frame, bg='white')
        block_frame.pack(fill='x', pady=5)
        
        self.block_size_scale = tk.Scale(block_frame, from_=1, to=5, orient='horizontal',
                                         variable=self.block_size_var, bg='white',
                                         font=('Arial', 9), length=200)
        self.block_size_scale.pack(side='left', fill='x', expand=True)
        tk.Label(block_frame, textvariable=self.block_size_var, bg='white', 
                font=('Arial', 9, 'bold'), width=3).pack(side='right', padx=5)
        
        tk.Label(frame, text="(1=detailed, 5=fast)", bg='white', 
                font=('Arial', 8), fg='gray').pack(anchor='w')
        
        # Diagonal Penalty
        tk.Label(frame, text="Diagonal Penalty:", bg='white', font=('Arial', 9)).pack(anchor='w', pady=(10,0))
        self.diagonal_penalty_var = tk.DoubleVar(value=8.0)
        penalty_frame = tk.Frame(frame, bg='white')
        penalty_frame.pack(fill='x', pady=5)
        
        self.penalty_scale = tk.Scale(penalty_frame, from_=1.0, to=20.0, resolution=0.5,
                                      orient='horizontal', variable=self.diagonal_penalty_var,
                                      bg='white', font=('Arial', 9), length=200)
        self.penalty_scale.pack(side='left', fill='x', expand=True)
        tk.Label(penalty_frame, textvariable=self.diagonal_penalty_var, bg='white',
                font=('Arial', 9, 'bold'), width=4).pack(side='right', padx=5)
        
        tk.Label(frame, text="(higher=straighter path)", bg='white',
                font=('Arial', 8), fg='gray').pack(anchor='w')
        
        # Line Width
        tk.Label(frame, text="Line Width:", bg='white', font=('Arial', 9)).pack(anchor='w', pady=(10,0))
        self.line_width_var = tk.IntVar(value=4)
        width_frame = tk.Frame(frame, bg='white')
        width_frame.pack(fill='x', pady=5)
        
        self.width_scale = tk.Scale(width_frame, from_=1, to=10, orient='horizontal',
                                    variable=self.line_width_var, bg='white',
                                    font=('Arial', 9), length=200)
        self.width_scale.pack(side='left', fill='x', expand=True)
        tk.Label(width_frame, textvariable=self.line_width_var, bg='white',
                font=('Arial', 9, 'bold'), width=3).pack(side='right', padx=5)
        
    def create_action_controls(self, parent):
        frame = tk.LabelFrame(parent, text="🎯 Actions", font=('Arial', 11, 'bold'),
                             bg='white', fg='#2c3e50', padx=10, pady=10)
        frame.pack(fill='x', padx=10, pady=10)
        
        self.select_btn = tk.Button(frame, text="Select Starting Point", 
                                    command=self.select_starting_point,
                                    bg='#DA3B01', fg='white', font=('Arial', 10, 'bold'),
                                    relief='raised', cursor='hand2', width=20, height=2,
                                    state='disabled', activebackground='#A82E0D', activeforeground='white', disabledforeground='white')
        self.select_btn.pack(pady=5)
        
        self.extract_btn = tk.Button(frame, text="Extract Path", 
                                     command=self.extract_path_threaded,
                                     bg='#6B46C1', fg='white', font=('Arial', 10, 'bold'),
                                     relief='raised', cursor='hand2', width=20, height=2,
                                     state='disabled', activebackground='#4C2896', activeforeground='white', disabledforeground='white')
        self.extract_btn.pack(pady=5)
        
        self.reset_btn = tk.Button(frame, text="Reset", command=self.reset_all,
                                   bg='#D13438', fg='white', font=('Arial', 10, 'bold'),
                                   relief='raised', cursor='hand2', width=20, height=2, activebackground='#A4373A', activeforeground='white', disabledforeground='white')
        self.reset_btn.pack(pady=5)
        
    def create_info_display(self, parent):
        frame = tk.LabelFrame(parent, text="ℹ️ Information", font=('Arial', 11, 'bold'),
                             bg='white', fg='#2c3e50', padx=10, pady=10)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.info_text = tk.Text(frame, height=15, width=35, bg='#f9f9f9',
                                font=('Courier', 9), relief='sunken', borderwidth=2,
                                wrap='word')
        self.info_text.pack(fill='both', expand=True)
        
        scrollbar = tk.Scrollbar(frame, command=self.info_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.info_text.config(yscrollcommand=scrollbar.set)
        
        self.log_message("Welcome! Load an image to start.")
        
    def create_image_display(self, parent):
        tk.Label(parent, text="Image Display", font=('Arial', 12, 'bold'),
                bg='white', fg='#2c3e50').pack(pady=10)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initialize with empty plots
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        
        self.ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        self.ax2.set_title('Result', fontsize=12, fontweight='bold')
        self.ax1.axis('off')
        self.ax2.axis('off')
        
        self.canvas.draw()
        
    def log_message(self, message):
        self.info_text.insert('end', f"{message}\n")
        self.info_text.see('end')
        self.root.update()
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.log_message(f"\n📂 Loading image...")
            img = Image.open(file_path)
            self.gray_img = img.convert('L')
            self.img_array = np.asarray(self.gray_img)
            
            # Display original image
            self.ax1.clear()
            self.ax1.imshow(self.gray_img, cmap='gray')
            self.ax1.set_title('Original Image', fontsize=12, fontweight='bold')
            self.ax1.axis('off')
            self.canvas.draw()
            
            self.log_message(f"✓ Image loaded successfully")
            self.log_message(f"  Shape: {self.img_array.shape}")
            self.log_message(f"  Range: [{self.img_array.min()}, {self.img_array.max()}]")
            
            # Enable buttons
            self.select_btn.config(state='normal')
            self.extract_btn.config(state='normal')
            
            # Reset start point
            self.start_row = None
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            self.log_message(f"❌ Error: {str(e)}")
            
    def select_starting_point(self):
        if self.img_array is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        self.log_message("\n🎯 Click on the vessel to select starting point...")
        
        # Clear previous selection
        self.ax1.clear()
        self.ax1.imshow(self.gray_img, cmap='gray')
        self.ax1.set_title('Click to select starting point', fontsize=12, fontweight='bold')
        self.ax1.axvline(x=self.img_array.shape[1] * 0.05, color='red', 
                        linestyle='--', linewidth=2, alpha=0.7, label='Recommended area')
        self.ax1.legend()
        self.ax1.axis('off')
        self.canvas.draw()
        
        # Connect click event
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def on_click(self, event):
        if event.inaxes == self.ax1:
            if event.xdata is not None and event.ydata is not None:
                self.start_row = int(event.ydata)
                
                # Draw marker
                self.ax1.plot(event.xdata, event.ydata, 'ro', markersize=10)
                self.ax1.plot(event.xdata, event.ydata, 'yo', markersize=5)
                self.canvas.draw()
                
                self.log_message(f"✓ Selected row: {self.start_row}")
                
                # Disconnect click event
                self.fig.canvas.mpl_disconnect(self.cid)
                
    def extract_path_threaded(self):
        # Run in separate thread to keep GUI responsive
        thread = threading.Thread(target=self.extract_path)
        thread.daemon = True
        thread.start()
        
    def extract_path(self):
        if self.img_array is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        try:
            # Disable buttons during processing
            self.root.after(0, lambda: self.extract_btn.config(state='disabled'))
            self.root.after(0, lambda: self.select_btn.config(state='disabled'))
            
            block_size = self.block_size_var.get()
            diagonal_penalty = self.diagonal_penalty_var.get()
            line_width = self.line_width_var.get()
            
            self.log_message(f"\n🔄 Starting extraction...")
            self.log_message(f"  Block size: {block_size}")
            self.log_message(f"  Diagonal penalty: {diagonal_penalty}")
            
            # Block averaging
            self.log_message("  Performing block averaging...")
            self.reduced_array, h_crop, w_crop = self.block_average_image(
                self.img_array, block_size=block_size
            )
            
            # Map starting row
            start_row_reduced = None
            if self.start_row is not None:
                start_row_reduced = self.start_row // block_size
                self.log_message(f"  Mapped start: row {self.start_row} → {start_row_reduced}")
            
            # Extract path
            self.log_message("  Extracting vessel path...")
            min_cost, path_reduced = self.extract_vessel_path(
                self.reduced_array,
                start_row=start_row_reduced,
                diagonal_penalty=diagonal_penalty
            )
            
            if not path_reduced:
                self.log_message("❌ No path found!")
                messagebox.showerror("Error", "No valid path found!")
                return
                
            self.log_message(f"  Path length: {len(path_reduced)} pixels")
            
            # Scale to original
            self.log_message("  Scaling to original size...")
            self.path_original = self.scale_path_to_original(path_reduced, block_size)
            
            # Visualize
            self.log_message("  Generating visualization...")
            self.result_img = self.visualize_path(self.img_array, self.path_original, line_width)
            
            # Display result
            self.root.after(0, self.display_result)
            
            self.log_message("✓ Extraction complete!")
            
            # Enable save button
            self.root.after(0, lambda: self.save_btn.config(state='normal'))
            
        except Exception as e:
            self.log_message(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Extraction failed:\n{str(e)}")
            
        finally:
            # Re-enable buttons
            self.root.after(0, lambda: self.extract_btn.config(state='normal'))
            self.root.after(0, lambda: self.select_btn.config(state='normal'))
            
    def display_result(self):
        self.ax2.clear()
        self.ax2.imshow(self.result_img)
        self.ax2.set_title('Extracted Vessel Path', fontsize=12, fontweight='bold')
        self.ax2.axis('off')
        self.canvas.draw()
        
    def save_result(self):
        if self.result_img is None:
            messagebox.showwarning("Warning", "No result to save!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            self.result_img.save(file_path)
            self.log_message(f"✓ Result saved to: {file_path}")
            messagebox.showinfo("Success", "Result saved successfully!")
            
    def reset_all(self):
        self.img_array = None
        self.gray_img = None
        self.result_img = None
        self.start_row = None
        self.reduced_array = None
        self.path_original = None
        
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        self.ax2.set_title('Result', fontsize=12, fontweight='bold')
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.canvas.draw()
        
        self.select_btn.config(state='disabled')
        self.extract_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        
        self.log_message("\n🔄 Reset complete. Load a new image to start.")
        
    # ========== Core Algorithm Methods ==========
    
    def block_average_image(self, image_array, block_size=3):
        h, w = image_array.shape
        h_cropped = (h // block_size) * block_size
        w_cropped = (w // block_size) * block_size
        cropped_arr = image_array[:h_cropped, :w_cropped]
        
        reduced_arr = cropped_arr.reshape(
            h_cropped // block_size, block_size,
            w_cropped // block_size, block_size
        ).mean(axis=(1, 3))
        
        return reduced_arr, h_cropped, w_cropped
        
    def extract_vessel_path(self, cost_map, start_row=None, diagonal_penalty=5.0):
        ROWS, COLS = cost_map.shape
        
        # Compute gradient
        grad_x = ndimage.sobel(cost_map, axis=1)
        grad_y = ndimage.sobel(cost_map, axis=0)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        gradient_norm = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8) * 255
        
        # Combined cost
        vessel_cost = 0.7 * cost_map + 0.3 * gradient_norm
        
        # DP table
        dp = np.full((ROWS, COLS), float('inf'))
        parent = np.full((ROWS, COLS), -1, dtype=int)
        
        # Base case
        if start_row is not None:
            dp[start_row, 0] = vessel_cost[start_row, 0]
        else:
            for r in range(ROWS):
                dp[r, 0] = vessel_cost[r, 0]
        
        # Fill DP table
        for c in range(1, COLS):
            for r in range(ROWS):
                current_pixel_cost = vessel_cost[r, c]
                best_cost = float('inf')
                best_parent = -1
                
                # From upper-left
                if r > 0 and dp[r-1, c-1] != float('inf'):
                    cost = dp[r-1, c-1] + current_pixel_cost + diagonal_penalty
                    if cost < best_cost:
                        best_cost = cost
                        best_parent = r - 1
                
                # From left
                if dp[r, c-1] != float('inf'):
                    cost = dp[r, c-1] + current_pixel_cost
                    if cost < best_cost:
                        best_cost = cost
                        best_parent = r
                
                # From lower-left
                if r < ROWS - 1 and dp[r+1, c-1] != float('inf'):
                    cost = dp[r+1, c-1] + current_pixel_cost + diagonal_penalty
                    if cost < best_cost:
                        best_cost = cost
                        best_parent = r + 1
                
                if best_cost != float('inf'):
                    dp[r, c] = best_cost
                    parent[r, c] = best_parent
        
        # Find endpoint
        end_row = np.argmin(dp[:, COLS-1])
        min_total_cost = dp[end_row, COLS-1]
        
        if min_total_cost == float('inf'):
            return min_total_cost, []
        
        # Backtrack
        path = []
        current_row = end_row
        for c in range(COLS - 1, -1, -1):
            path.append((current_row, c))
            if c > 0:
                current_row = parent[current_row, c]
        
        path.reverse()
        return min_total_cost, path
        
    def scale_path_to_original(self, path, block_size):
        scaled_path = [(r * block_size + block_size // 2, 
                        c * block_size + block_size // 2) 
                       for r, c in path]
        return scaled_path
        
    def visualize_path(self, image_array, path, line_width=3):
        result_img = Image.fromarray(image_array.astype(np.uint8), mode='L')
        result_img_rgb = result_img.convert('RGB')
        draw = ImageDraw.Draw(result_img_rgb)
        
        if len(path) > 1:
            pil_path = [(col, row) for row, col in path]
            draw.line(pil_path, fill=(255, 255, 0), width=line_width+2)  # Yellow outline
            draw.line(pil_path, fill=(255, 0, 0), width=line_width)      # Red center
        
        return result_img_rgb


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = VesselPathExtractorGUI(root)
    root.mainloop()
