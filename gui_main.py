import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import sys
import os

try:
    from agent_main import AutonomousElectrodeAgent
except ImportError as e:
    print(f"CRITICAL: Cannot import agent_main.py. Check component files.")
    sys.exit(1)


class AgentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neuro-Electrode Optimization AI Agent")
        self.root.geometry("1100x750")
        
        self.status_ok = False
        self._init_agent()
        
        self.mri_path = None
        if self.status_ok:
            self._setup_ui()

    def _init_agent(self):
        """Safely initializes the backend agent."""
        try:
            self.agent = AutonomousElectrodeAgent("config.json")
            self.status_ok = True
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Could not start Agent.\nDetails: {str(e)}\n\nEnsure all dependency files are present.")
            self.root.destroy()
            sys.exit()

    def _setup_ui(self):
        # Style
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 10))
        
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50")
        header_frame.pack(fill="x")
        lbl_header = tk.Label(header_frame, text="Neuro-Electrode Geometry Optimization AI Agent", 
                              font=("Segoe UI", 16, "bold"), bg="#2c3e50", fg="white", pady=12)
        lbl_header.pack()
        
        # Main Container
        main_frame = tk.Frame(self.root, padx=15, pady=15)
        main_frame.pack(fill="both", expand=True)
        
        # Left Column: Controls
        left_col = tk.Frame(main_frame, width=320)
        left_col.pack(side="left", fill="y", padx=(0, 10))
        
        # 1. Input Group
        grp_input = tk.LabelFrame(left_col, text="Patient & Scan Data", font=("Segoe UI", 10, "bold"), fg="#34495e")
        grp_input.pack(fill="x", pady=5, ipady=5)
        
        tk.Label(grp_input, text="Patient ID:").pack(anchor="w", padx=10)
        self.ent_pid = tk.Entry(grp_input)
        self.ent_pid.insert(0, "Subject_MRI")
        self.ent_pid.pack(fill="x", padx=10, pady=(0, 5))
        
        tk.Button(grp_input, text="Load MRI (.nii, .nii.gz)", command=self._browse_file, bg="#ecf0f1").pack(fill="x", padx=10, pady=5)
        self.lbl_file = tk.Label(grp_input, text="No file selected (Using Synthetic Data)", fg="gray", font=("Arial", 8, "italic"), wraplength=280)
        self.lbl_file.pack(pady=2)

        # 2. Parameters Group
        grp_param = tk.LabelFrame(left_col, text="Surgical Parameters", font=("Segoe UI", 10, "bold"), fg="#34495e")
        grp_param.pack(fill="x", pady=10, ipady=5)
        
        tk.Label(grp_param, text="Surgical Target Depth (mm):").pack(anchor="w", padx=10)
        self.ent_disp = tk.Entry(grp_param)
        self.ent_disp.insert(0, "120.0")
        self.ent_disp.pack(fill="x", padx=10, pady=(0, 5))
        tk.Label(grp_param, text="Note: AI may override this with anatomical depth.", fg="darkgray", font=("Arial", 7)).pack(anchor="w", padx=10)


        # 3. Run Button
        self.btn_run = tk.Button(left_col, text="START AUTONOMOUS ANALYSIS", bg="#27ae60", fg="white", 
                                 font=("Segoe UI", 11, "bold"), relief="flat", command=self._start_thread)
        self.btn_run.pack(fill="x", pady=15)

        # 4. System Log
        tk.Label(left_col, text="Agent Reasoning Log:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.txt_log = tk.Text(left_col, height=18, width=40, font=("Consolas", 9), bg="#f8f9fa", relief="solid", bd=1)
        self.txt_log.pack(fill="both", expand=True)

        # Right Column: Visualization 
        right_col = tk.Frame(main_frame, bg="white", relief="groove", bd=1)
        right_col.pack(side="right", fill="both", expand=True)
        
        self.viz_frame = tk.Frame(right_col, bg="white")
        self.viz_frame.pack(fill="both", expand=True)
        
        # Placeholder Text
        self.lbl_placeholder = tk.Label(self.viz_frame, text="Visualization Pipeline (3 Steps):\n1. T1w Volume\n2. AI Segmentation (Probability Map)\n3. Optimized Electrode Geometry", 
                                        bg="white", fg="#bdc3c7", font=("Segoe UI", 14))
        self.lbl_placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def _browse_file(self):
        fn = filedialog.askopenfilename(filetypes=[("NIfTI/Neuroimaging", "*.nii *.nii.gz")])
        if fn:
            self.mri_path = fn
            short_name = fn.split('/')[-1]
            self.lbl_file.config(text=f"Loaded: {short_name}", fg="#2980b9")

    def _start_thread(self):
        self.btn_run.config(state="disabled", text="Processing... (Please Wait)")
        self.txt_log.delete(1.0, tk.END)
        self.txt_log.insert(tk.END, "Initializing AI Pipeline...\n")
        
        t = threading.Thread(target=self._run_analysis)
        t.start()

    def _run_analysis(self):
        """Executes the agent in a separate thread."""
        try:
            pid = self.ent_pid.get()
            try:
                disp = float(self.ent_disp.get()) 
            except ValueError:
                self._log("ERROR: Invalid Depth value. Using default 120.0 mm.")
                disp = 120.0

            self._log(f"Loading MRI/Config resources...")
            results = self.agent.run_autonomous_analysis(pid, disp, self.mri_path)
            
            if not results:
                self._log("ERROR: Analysis returned no results.")
                return

            brain = results.get('brain_analysis', {})
            rec = results.get('final_recommendations', {})
            
            self._log("\n>> MULTIMODAL ANALYSIS SUMMARY")
            self._log(f"Target Region: {brain.get('identified_regions', ['Unknown'])[0]}")
            self._log(f"AI Confidence: {brain.get('target_region_confidence', 0.0):.2f}")
            self._log(f"Tissue Density: {brain.get('tissue_density_score', 0.0):.3f}")
            self._log(f"AI Target Depth: {brain.get('target_center_depth_mm', disp):.1f} mm")

            self._log("\n>> OPTIMIZATION RESULTS")
            self._log(f"Target Current: {rec.get('target_used_uA', 0.0)} uA")
            self._log(f"Rec. Area: {rec.get('optimized_area_um2', 0.0)} μm²")
            self._log(f"Rec. Pitch: {rec.get('optimized_pitch_um', 0.0)} μm")
            self._log(f"Predicted Output: {rec.get('predicted_threshold_uA', 0.0)} uA")
            self._log(f"Error (Max {self.agent.config['safety_thresholds']['max_error_uA']}uA): {rec.get('current_error_uA', 0.0)} uA")
            
            self.root.after(0, lambda: self._update_viz(results))
            
        except Exception as e:
            self._log(f"\nCRITICAL ERROR: {str(e)}")
        finally:
            self.root.after(0, self._reset_ui)

    def _log(self, msg):
        self.root.after(0, lambda: self.txt_log.insert(tk.END, msg + "\n"))
        self.root.after(0, lambda: self.txt_log.see(tk.END))

    def _reset_ui(self):
        self.btn_run.config(state="normal", text="START AUTONOMOUS ANALYSIS")

    def _update_viz(self, results):
        """Creates the 3-panel visualization for Volume, Segmentation, and Geometry."""
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
            
        rec = results.get('final_recommendations', {})
        brain = results.get('brain_analysis', {})
        mri_data = results.get('mri_data', {}) 
        
        # Data validation 
        if not all([mri_data, rec]):
            self.lbl_placeholder = tk.Label(self.viz_frame, text="Visualization Data Missing.", bg="white", fg="red")
            self.lbl_placeholder.place(relx=0.5, rely=0.5, anchor="center")
            return

        # 1. Extract Metrics
        area = rec.get('optimized_area_um2', 2500.0)
        pitch = rec.get('optimized_pitch_um', 500.0)
        side = area**0.5
        target_region = brain.get('identified_regions', ['Unknown'])[0]
        target_depth_mm = brain.get('target_center_depth_mm', 120.0)
        
        # 2. Setup Figure with Subplots (1x3 Layout)
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        fig.subplots_adjust(wspace=0.3)
        
        # SUBPLOT 1: T1 Volume & Target Slice
        vol = mri_data.get('full_volume', np.zeros((10,10,10)))
        
        # Safety check for volume dimensions
        if vol.ndim < 3 or vol.shape[2] < 1:
            central_slice_idx = 0
            # Use a dummy 2D slice if volume is corrupt
            slice_2d = np.zeros((50, 50)) 
        else:
            central_slice_idx = vol.shape[2] // 2
            slice_2d = vol[:, :, central_slice_idx]
            
        axs[0].imshow(slice_2d, cmap='gray')
        axs[0].set_title(f"T1w Volume Slice (Axial)", fontsize=9, fontweight='bold')
        axs[0].set_axis_off()
        
        # Simulate target indication
        center_y, center_x = slice_2d.shape[0] / 2, slice_2d.shape[1] / 2
        axs[0].plot(center_x, center_y, 'y*', markersize=10, markeredgecolor='black', label=target_region)
        axs[0].legend(loc='lower left', fontsize=7, framealpha=0.7)

        # SUBPLOT 2: Segmentation (White Matter Probability Map) 
        seg_mask = mri_data.get('segmentation_mask', np.zeros(vol.shape))
        if seg_mask.ndim >= 3 and seg_mask.shape[2] > central_slice_idx:
            axial_seg = seg_mask[:, :, central_slice_idx]
        else:
            axial_seg = np.zeros(slice_2d.shape)
            
        axs[1].imshow(slice_2d, cmap='gray', alpha=0.5) # Background T1
        axs[1].imshow(axial_seg, cmap='jet', alpha=0.4) # Overlay WMPM
        axs[1].set_title(f"Probability Map for {target_region}", fontsize=9, fontweight='bold', color='#16a085')
        axs[1].set_axis_off()
        axs[1].text(5, 5, f"Target Depth: {target_depth_mm:.1f}mm", color='white', fontsize=8, backgroundcolor='#2c3e50')

        # SUBPLOT 3: Optimized Electrode Geometry 
        # Anode 
        rect1 = plt.Rectangle((0, 0), side, side, color='#e74c3c', alpha=0.8, label='Anode')
        axs[2].add_patch(rect1)
        # Cathode 
        rect2 = plt.Rectangle((0, side + pitch), side, side, color='#3498db', alpha=0.8, label='Cathode')
        axs[2].add_patch(rect2)
        axs[2].text(side + 10, side/2, f"Area:{area:.0f}μm²", verticalalignment='center', fontsize=8)
        axs[2].text(side/2, side + (pitch/2), f"Pitch:{pitch:.0f}μm", horizontalalignment='center', verticalalignment='center', fontsize=8, color='green')
        total_height = (side * 2) + pitch
        margin = max(total_height, side, 10) * 0.3
        if total_height > 0:
            axs[2].set_xlim(-margin, side + margin)
            axs[2].set_ylim(-(side * 0.2), total_height + margin)
        else:
             axs[2].set_xlim(-10, 100)
             axs[2].set_ylim(-10, 100)
             
        axs[2].set_aspect('equal')
        axs[2].set_title(f"3. Optimized Electrode\nPred: {rec.get('predicted_threshold_uA', 0.0)}uA | Target: {rec.get('target_used_uA', 0.0)}uA", fontsize=9, fontweight='bold')
        axs[2].set_xlabel("Width (μm)")
        axs[2].set_ylabel("Displacement Axis (μm)")
        axs[2].legend(loc='upper right')

        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig) 

if __name__ == "__main__":
    root = tk.Tk()
    app = AgentGUI(root)
    root.mainloop()