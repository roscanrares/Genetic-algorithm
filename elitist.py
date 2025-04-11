import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_right


def main():
    # Input
    print("\nDomeniul de definitie ")
    a = float(input("Capatul stang "))
    b = float(input("Capatul drept "))

    print("\nCoeficientii functiei ")
    c2 = float(input("c2  "))
    if c2 > 0:
        return 0

    c1 = float(input("c1 "))
    c0 = float(input("c0 "))

    print("\nParametri algoritm genetic")
    population_size = int(input("Dimensiune populatie "))
    if population_size < 4:
        return 0

    precision = int(input("Precizie ")) # 1-10
    if precision < 1:
        return 0

    crossover_prob = float(input("Probabilitate recombinare "))
    if crossover_prob < 0 or crossover_prob > 1:
        return 0

    mutation_rate = float(input("Probabilitate mutatie "))
    if mutation_rate < 0 or mutation_rate > 1:
        return 0

    generations = int(input("Numar generatii "))
    if generations < 1:
        return 0

    # Calcule preliminare
    num_intervals = (b - a) * (10 ** precision)
    l = int(np.ceil(np.log2(num_intervals)))
    print(f"\nLungimea cromozomului calculata: {l} biti")

    # Functii auxiliare
    def decode(chromosome):
        x_int = int(chromosome, 2)
        return a + (b - a) * x_int / (2 ** l - 1)

    def fitness(x):
        return c4 * (x ** 4) + c3 * (x ** 3) + c2 * (x ** 2) + c1 * x + c0

    # Initializare populatie
    population = []
    for _ in range(population_size):
        random_binary_string = ''.join(np.random.choice(['0', '1'], size=l))
        population.append(random_binary_string)

    # Pentru vizualizare grafica
    max_fitness_history = []
    mean_fitness_history = []

    # Fisier de iesire
    with open('Evolutie_Elitism.txt', 'w') as f:
        f.write("=== Algoritm Genetic pentru Maximizare ===\n")
        f.write(f"Functia: {c2}x^2 + {c1}x + {c0}\n")
        f.write(f"Domeniu: [{a}, {b}]\n")
        f.write(f"Parametri: Populatie={population_size}, Precizie={precision}, ")
        f.write(f"Recombinare={crossover_prob}, Mutatie={mutation_rate}, Generatii={generations}\n\n")

        for gen in range(generations):
            # Decodificare si calcul fitness
            decoded = [decode(chrom) for chrom in population]
            fitness_scores = [fitness(x) for x in decoded]

            # Selectam max
            elite_idx = np.argmax(fitness_scores)
            elite = population[elite_idx]

            # Scriere detalii generatie
            f.write(f"\n=== Generatia {gen + 1} ===\n")
            f.write(f"Individul elitist: {elite} cu fitness {fitness_scores[elite_idx]:.{precision}f}\n")

            # Probabilitati de selectie
            total_fitness = sum(fitness_scores)
            probabilities = [f / total_fitness for f in fitness_scores]
            cumulative_probs = np.cumsum(probabilities)

            f.write("\nProbabilitati de selectie:\n")
            for i, p in enumerate(probabilities):
                f.write(f"Cromozom {i + 1}: p={p:.6f} q={cumulative_probs[i]:.6f}\n")

            # Selectie
            f.write("\nProcesul de selectie:\n")
            selected_parents = []
            for _ in range(population_size - 1):
                u = np.random.uniform(0, 1)
                idx = bisect_right(cumulative_probs, u)
                selected_parents.append(population[idx])
                f.write(f"u={u:.6f} -> selectat cromozomul {idx + 1}\n")

            # Recombinare
            f.write("\nRecombinare:\n")
            next_population = []
            for i in range(0, len(selected_parents) - 1, 2):
                parent1, parent2 = selected_parents[i], selected_parents[i + 1]
                if np.random.rand() < crossover_prob:
                    point = np.random.randint(1, l)
                    child1 = parent1[:point] + parent2[point:]
                    child2 = parent2[:point] + parent1[point:]
                    f.write(f"Recombinare intre {i + 1} si {i + 2} la punctul {point}: "
                            f"{parent1} + {parent2} -> {child1} + {child2}\n")
                    next_population.extend([child1, child2])
                else:
                    f.write(f"Fara recombinare pentru perechea {i + 1}-{i + 2}\n")
                    next_population.extend([parent1, parent2])

            # # Mutatie
            f.write("\nMutatie:\n")
            mutated_population = []
            for chrom in next_population:
                new_chrom = list(chrom)
                mutated = False
                for j in range(l):
                    if np.random.rand() < mutation_rate:
                        new_chrom[j] = '1' if new_chrom[j] == '0' else '0'
                        mutated = True
                if mutated:
                    f.write(f"{chrom} -> {''.join(new_chrom)}\n")
                mutated_population.append(''.join(new_chrom))

            population = [elite] + mutated_population[:population_size - 1]

            # Calcul final
            decoded = [decode(chrom) for chrom in population]
            fitness_scores = [fitness(x) for x in decoded]
            max_fitness = max(fitness_scores)
            mean_fitness = np.mean(fitness_scores)

            max_fitness_history.append(max_fitness)
            mean_fitness_history.append(mean_fitness)

            if gen > 0:
                f.write(f"\nRezumat generatie {gen + 1}:\n")
                f.write(f"Max Fitness = {max_fitness:.6f}\n")
                f.write(f"Mean Fitness = {mean_fitness:.6f}\n")

        # Rezultat
        best_idx = np.argmax(fitness_scores)
        best_x = decoded[best_idx]
        best_fitness = fitness_scores[best_idx]

        f.write("\n=== REZULTAT FINAL ===\n")
        f.write(f"Maximul functiei: f({best_x:.{precision}f}) = {best_fitness:.{precision}f}\n")

    # Vizualizare
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, generations + 1), max_fitness_history, label='Max Fitness')
    plt.plot(range(1, generations + 1), mean_fitness_history, label='Mean Fitness')
    plt.xlabel('Generatie')
    plt.ylabel('Fitness')
    plt.title(f"Evolutia fitness-ului\nf(x) = {c2}x² + {c1}x + {c0}")
    plt.legend()
    plt.grid()
    plt.savefig('Evolutie_Elitism.png')
    plt.show()

    print("\nExecutie completaa! Rezultatele au fost salvate in:")
    print("- Evolutie_Elitism.txt (detalii complete ale algoritmului)")
    print("- Evolutie_Elitism.png (grafic evolutie fitness)")


if __name__ == "__main__":
    main()