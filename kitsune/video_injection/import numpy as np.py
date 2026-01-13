import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# --- Paramètres du modèle (D'après l'énoncé et vos notes) ---
n = 1024        # Taille totale du paquet en bits
k = 990         # On suppose k bits utiles (le reste est le CRC)
M_ARQ = 3       # Nombre maximal de retransmissions
N_paquets = 500 # Nombre de paquets simulés (Supérieur à 156 pour lisser)

# Plage de SNR (Eb/N0) à tester
SNR_dB_range = np.arange(0, 26, 2) 

# Tableaux pour stocker les résultats
debit_simu = []
per_simu = []
debit_theo = []
per_theo = []

# --- Fonctions Utiles ---

def get_ber_rayleigh_inst(snr_lin):
    """Calcule la proba d'erreur binaire pour un tirage de canal donné."""
    # Simulation du canal h (Rayleigh)
    h_real = np.random.randn()
    h_imag = np.random.randn()
    # Puissance instantanée du canal |h|^2
    gain_canal = (h_real**2 + h_imag**2) / 2
    
    # Proba d'erreur bit (BPSK) pour ce canal spécifique
    # Pb = Q(sqrt(2 * |h|^2 * Eb/N0)) -> Q(x) = 0.5*erfc(x/sqrt(2))
    return 0.5 * erfc(np.sqrt(gain_canal * snr_lin))

def ber_theorique_moyen(snr_lin):
    """Formule théorique du BER moyen en Rayleigh (pour comparaison)."""
    return 0.5 * (1 - np.sqrt(snr_lin / (1 + snr_lin)))

print(f"Simulation lancée avec M_ARQ={M_ARQ} sur {N_paquets} paquets...")

# --- Boucle Principale (Algorithme d'estimation) ---

for snr_db in SNR_dB_range:
    snr_lin = 10**(snr_db / 10)
    
    # Compteurs pour la simulation
    nb_paquets_perdus = 0
    nb_bits_utiles_recus = 0
    nb_total_transmissions = 0
    
    for _ in range(N_paquets):
        # Début algo Stop-and-Wait pour UN paquet
        succes = False
        m_essai = 0
        
        while m_essai <= M_ARQ and not succes:
            m_essai += 1
            nb_total_transmissions += 1
            
            # 1. Tirage du canal (nouvelle réalisation à chaque envoi)
            pe_inst = get_ber_rayleigh_inst(snr_lin)
            
            # 2. Vérification Erreur (Simulation CRC)
            # Proba qu'au moins 1 bit soit faux dans le paquet de n bits
            prob_paquet_errone = 1 - (1 - pe_inst)**n
            
            # Tirage aléatoire : Erreur détectée ?
            if np.random.rand() > prob_paquet_errone:
                succes = True # ACK = 1
            else:
                pass # ACK = 0, on boucle (retransmission)
        
        # Fin de la boucle de retransmission
        if succes:
            nb_bits_utiles_recus += k
        else:
            nb_paquets_perdus += 1 # Echec après M retransmissions

    # --- Calcul des résultats Simulation ---
    # Débit = Bits utiles / (Nombre total d'utilisations du canal * taille paquet)
    eta = nb_bits_utiles_recus / (nb_total_transmissions * n)
    per = nb_paquets_perdus / N_paquets
    
    debit_simu.append(eta)
    per_simu.append(per)

    # --- Calculs Théoriques (Formules du I) ---
    # On utilise le BER moyen théorique
    ber_moy = ber_theorique_moyen(snr_lin)
    Pd = 1 - (1 - ber_moy)**n # Proba erreur paquet moyenne
    
    # PER théorique = Pd^(M+1)
    per_th = Pd**(M_ARQ + 1)
    per_theo.append(per_th)
    
    # Débit théorique (approx Stop-and-Wait tronqué)
    nb_trans_moy = (1 - Pd**(M_ARQ + 1)) / (1 - Pd)
    eta_th = (k/n) * (1/nb_trans_moy) * (1 - per_th)
    debit_theo.append(eta_th)

# --- Affichage des courbes ---
plt.figure(figsize=(12, 5))

# Plot Débit
plt.subplot(1, 2, 1) 
plt.plot(SNR_dB_range, debit_simu, 'bo-', label='Simulation')
plt.plot(SNR_dB_range, debit_theo, 'r--', label='Théorie')
plt.xlabel('SNR Moy (Eb/N0) [dB]')
plt.ylabel('Débit (Throughput)')
plt.title(f'Débit Moyen (M={M_ARQ})')
plt.grid(True)
plt.legend()

# Plot PER
plt.subplot(1, 2, 2) 
plt.semilogy(SNR_dB_range, per_simu, 'bo-', label='Simulation')
plt.semilogy(SNR_dB_range, per_theo, 'r--', label='Théorie')
plt.xlabel('SNR Moy (Eb/N0) [dB]')
plt.ylabel('PER (Log scale)')
plt.title('Taux Erreur Paquet')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()